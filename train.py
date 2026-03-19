import argparse
import random
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch import optim
from tqdm import tqdm

from layers import hours, resolve_device
from load import max_len
from models import Model


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint_records(checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    records = checkpoint.get("records", {})
    if not records or not records.get("epoch"):
        raise ValueError(f"No training records found in checkpoint: {checkpoint_path}")
    return checkpoint, records


def plot_records(records, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting. Install it with `pip install matplotlib`.") from exc

    epochs = records["epoch"]
    valid = np.asarray(records["acc_valid"], dtype=np.float64)
    test = np.asarray(records["acc_test"], dtype=np.float64)
    ks = (1, 5, 10, 20)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for idx, k in enumerate(ks):
        axes[0].plot(epochs, valid[:, idx], marker="o", linewidth=1.5, label=f"Recall@{k}")
        axes[1].plot(epochs, test[:, idx], marker="o", linewidth=1.5, label=f"Recall@{k}")

    axes[0].set_title("Validation Recall Curves")
    axes[1].set_title("Test Recall Curves")
    axes[1].set_xlabel("Epoch")
    axes[0].set_ylabel("Recall")
    axes[1].set_ylabel("Recall")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()
    axes[1].legend()

    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_records_from_checkpoint(checkpoint_path, device, output_path=None):
    _, records = load_checkpoint_records(checkpoint_path, device)
    target_path = output_path or (Path(checkpoint_path).with_suffix("").as_posix() + "_records.png")
    return plot_records(records, target_path)


def calculate_recall(prob, label, ks=(1, 5, 10, 20)):
    label = label.view(-1, 1)
    topk = torch.topk(prob, k=max(ks), dim=1).indices
    recalls = []
    for k in ks:
        hit = (topk[:, :k] == label).any(dim=1).float().sum().item()
        recalls.append(hit)
    return np.array(recalls, dtype=np.float64)


def sampling_prob(prob, label, num_neg):
    device = prob.device
    num_label, loc_count = prob.shape
    label = label.view(-1)

    candidate_mask = torch.ones(loc_count, dtype=torch.bool, device=device)
    candidate_mask[label] = False
    available_neg = torch.nonzero(candidate_mask, as_tuple=False).view(-1)
    if available_neg.numel() == 0:
        sampled_prob = prob.index_select(1, label)
        sampled_label = torch.arange(num_label, device=device)
        return sampled_prob, sampled_label

    num_neg = min(num_neg, int(available_neg.numel()))
    rand_perm = torch.randperm(available_neg.numel(), device=device)[:num_neg]
    neg_index = available_neg.index_select(0, rand_perm)
    sample_index = torch.cat([label, neg_index], dim=0)
    sampled_prob = prob.index_select(1, sample_index)
    sampled_label = torch.arange(num_label, device=device)
    return sampled_prob, sampled_label


class DataSet(data.Dataset):
    def __init__(self, traj, m1, v, label, length, device):
        self.traj = traj
        self.mat1 = m1
        self.vec = v
        self.label = label
        self.length = length
        self.device = device

    def __getitem__(self, index):
        return (
            self.traj[index].to(self.device),
            self.mat1[index].to(self.device),
            self.vec[index].to(self.device),
            self.label[index].to(self.device),
            self.length[index].to(self.device),
        )

    def __len__(self):
        return len(self.traj)


class Trainer:
    def __init__(self, model, records, tensors, args):
        self.model = model.to(args.device)
        self.records = records
        self.device = args.device
        self.start_epoch = records["epoch"][-1] + 1 if records["epoch"] else 1
        self.num_neg = args.num_neg
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.num_epoch = args.epochs
        self.threshold = np.mean(records["acc_valid"][-1]) if records["acc_valid"] else float("-inf")
        self.early_stop_patience = args.early_stop_patience
        self.early_stop_min_delta = args.early_stop_min_delta
        self.no_improve_count = 0
        self.save_path = Path(args.checkpoint)
        self.data_loader = data.DataLoader(
            dataset=DataSet(*tensors, device=args.device),
            batch_size=self.batch_size,
            shuffle=False,
        )
        self.part = len(tensors[0])
        self.mat2s = args.mat2s
        self.semantic = args.semantic
        self.social = args.social

    def evaluate(self, batch_size=1):
        return evaluate_model(
            self.model,
            tensors=(self.data_loader.dataset.traj, self.data_loader.dataset.mat1, self.mat2s,
                     self.data_loader.dataset.vec, self.semantic, self.social,
                     self.data_loader.dataset.label + 1, self.data_loader.dataset.length),
            device=self.device,
            batch_size=batch_size,
        )

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1)

        start_time = time.time()
        for epoch_idx in range(self.num_epoch):
            self.model.train()
            bar = tqdm(total=self.part, desc=f"epoch {self.start_epoch + epoch_idx}")
            for item in self.data_loader:
                person_input, person_m1, person_m2t, person_label, person_traj_len = item

                input_mask = torch.zeros((self.batch_size, max_len, 3), dtype=torch.long, device=self.device)
                m1_mask = torch.zeros(
                    (self.batch_size, max_len, max_len, 2), dtype=torch.float32, device=self.device
                )
                full_len = int(person_traj_len[0].item())

                for mask_len in range(1, full_len + 1):
                    input_mask[:, :mask_len] = 1
                    m1_mask[:, :mask_len, :mask_len] = 1

                    train_input = person_input * input_mask
                    train_m1 = person_m1 * m1_mask
                    train_m2t = person_m2t[:, mask_len - 1]
                    train_label = person_label[:, mask_len - 1]
                    train_len = torch.full((self.batch_size,), mask_len, dtype=torch.long, device=self.device)

                    prob = self.model(train_input, train_m1, self.mat2s, train_m2t, train_len, self.semantic, self.social)

                    if mask_len <= full_len - 2:
                        prob_sample, label_sample = sampling_prob(prob, train_label, self.num_neg)
                        loss_train = F.cross_entropy(prob_sample, label_sample)
                        loss_train.backward()
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()

                bar.update(self.batch_size)
            bar.close()

            metrics = self.evaluate(batch_size=1)
            acc_valid = metrics["valid"]
            acc_test = metrics["test"]
            elapsed = time.time() - start_time
            print(
                f"epoch:{self.start_epoch + epoch_idx}, time:{elapsed:.2f}, "
                f"valid_recall@5:{acc_valid[1]:.4f}, valid_recall@10:{acc_valid[2]:.4f}"
            )
            print(
                f"epoch:{self.start_epoch + epoch_idx}, time:{elapsed:.2f}, "
                f"test_recall@5:{acc_test[1]:.4f}, test_recall@10:{acc_test[2]:.4f}"
            )

            self.records["acc_valid"].append(acc_valid)
            self.records["acc_test"].append(acc_test)
            self.records["epoch"].append(self.start_epoch + epoch_idx)

            current_score = float(np.mean(acc_valid))
            improved = current_score > self.threshold + self.early_stop_min_delta
            if improved:
                self.threshold = current_score
                self.no_improve_count = 0
                torch.save(
                    {
                        "state_dict": self.model.state_dict(),
                        "records": self.records,
                        "time": elapsed,
                        "device": str(self.device),
                    },
                    self.save_path,
                )
            else:
                self.no_improve_count += 1

            if self.early_stop_patience > 0 and self.no_improve_count >= self.early_stop_patience:
                print(
                    f"early_stop: epoch={self.start_epoch + epoch_idx}, "
                    f"best_valid_mean={self.threshold:.4f}, patience={self.early_stop_patience}"
                )
                break


def evaluate_model(model, tensors, device, batch_size=1):
    trajs, mat1, mat2s, mat2t, semantic, social, labels, lens = tensors
    loader = data.DataLoader(
        dataset=DataSet(trajs, mat1, mat2t, labels - 1, lens, device=device),
        batch_size=batch_size,
        shuffle=False,
    )

    valid_size, test_size = 0, 0
    acc_valid = np.zeros(4, dtype=np.float64)
    acc_test = np.zeros(4, dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for item in loader:
            person_input, person_m1, person_m2t, person_label, person_traj_len = item
            current_batch = person_input.shape[0]

            input_mask = torch.zeros((current_batch, max_len, 3), dtype=torch.long, device=device)
            m1_mask = torch.zeros((current_batch, max_len, max_len, 2), dtype=torch.float32, device=device)

            for row_idx in range(current_batch):
                full_len = int(person_traj_len[row_idx].item())
                for mask_len in range(1, full_len + 1):
                    input_mask[row_idx].zero_()
                    m1_mask[row_idx].zero_()
                    input_mask[row_idx, :mask_len] = 1
                    m1_mask[row_idx, :mask_len, :mask_len] = 1

                    eval_input = person_input[row_idx : row_idx + 1] * input_mask[row_idx : row_idx + 1]
                    eval_m1 = person_m1[row_idx : row_idx + 1] * m1_mask[row_idx : row_idx + 1]
                    eval_m2t = person_m2t[row_idx : row_idx + 1, mask_len - 1]
                    eval_label = person_label[row_idx : row_idx + 1, mask_len - 1]
                    eval_len = torch.full((1,), mask_len, dtype=torch.long, device=device)

                    prob = model(eval_input, eval_m1, mat2s, eval_m2t, eval_len, semantic, social)
                    if mask_len == full_len - 1:
                        valid_size += 1
                        acc_valid += calculate_recall(prob, eval_label)
                    elif mask_len == full_len:
                        test_size += 1
                        acc_test += calculate_recall(prob, eval_label)

    model.train()
    return {
        "valid": acc_valid / max(valid_size, 1),
        "test": acc_test / max(test_size, 1),
        "valid_size": valid_size,
        "test_size": test_size,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train STAN on a processed dataset.")
    parser.add_argument("--dataset", default="NYC")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--part", type=int, default=100, help="Number of users to train on. Use -1 for all users.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--num-neg", type=int, default=10)
    parser.add_argument("--embed-dim", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop training if validation mean recall does not improve for N consecutive epochs. 0 disables early stopping.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation mean recall improvement required to reset early stopping patience.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--plot-records", action="store_true", help="Load records from checkpoint and save recall curves.")
    parser.add_argument("--plot-output", default=None, help="Output image path for plotted records.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.device = resolve_device(args.device)
    set_random_seed(args.seed)
    checkpoint_path = args.checkpoint or f"best_stan_{args.dataset}.pth"
    args.checkpoint = checkpoint_path

    if args.plot_records:
        output_path = plot_records_from_checkpoint(checkpoint_path, args.device, args.plot_output)
        print(f"records_plot:{output_path}")
        return

    data_path = Path(args.data_dir) / f"{args.dataset}_data.pkl"
    with data_path.open("rb") as handle:
        file_data = joblib.load(handle)

    if len(file_data) == 11:
        trajs, mat1, mat2s, mat2t, semantic, social, labels, lens, u_max, l_max, _meta = file_data
    else:
        trajs, mat1, mat2s, mat2t, semantic, social, labels, lens, u_max, l_max = file_data
    mat1 = torch.as_tensor(mat1, dtype=torch.float32)
    mat2s = torch.as_tensor(mat2s, dtype=torch.float32, device=args.device)
    mat2t = torch.as_tensor(mat2t, dtype=torch.float32)
    semantic = torch.as_tensor(semantic, dtype=torch.float32, device=args.device)
    social = torch.as_tensor(social, dtype=torch.float32, device=args.device)
    lens = torch.as_tensor(lens, dtype=torch.long)

    if args.part > 0:
        trajs = trajs[: args.part]
        mat1 = mat1[: args.part]
        mat2t = mat2t[: args.part]
        labels = labels[: args.part]
        lens = lens[: args.part]

    ex = (mat1[:, :, :, 0].max(), mat1[:, :, :, 0].min(), mat1[:, :, :, 1].max(), mat1[:, :, :, 1].min())
    model = Model(t_dim=hours + 1, l_dim=l_max + 1, u_dim=u_max + 1, embed_dim=args.embed_dim, ex=ex, dropout=0)

    records = {"epoch": [], "acc_valid": [], "acc_test": []}
    if args.resume and Path(checkpoint_path).exists():
        checkpoint, records = load_checkpoint_records(checkpoint_path, args.device)
        model.load_state_dict(checkpoint["state_dict"])

    trainer = Trainer(
        model,
        records,
        tensors=(trajs, mat1, mat2t, labels - 1, lens),
        args=argparse.Namespace(**vars(args), mat2s=mat2s, semantic=semantic, social=social),
    )
    trainer.train()

    best_idx = int(np.argmax([np.mean(item) for item in trainer.records["acc_valid"]]))
    best_epoch = trainer.records["epoch"][best_idx]
    best_valid = trainer.records["acc_valid"][best_idx]
    best_test = trainer.records["acc_test"][best_idx]
    print(
        f"best_epoch:{best_epoch}, valid_recall@5:{best_valid[1]:.4f}, "
        f"valid_recall@10:{best_valid[2]:.4f}"
    )
    print(
        f"best_epoch:{best_epoch}, test_recall@5:{best_test[1]:.4f}, "
        f"test_recall@10:{best_test[2]:.4f}"
    )

    if Path(checkpoint_path).exists():
        output_path = plot_records_from_checkpoint(checkpoint_path, args.device, args.plot_output)
        print(f"records_plot:{output_path}")


if __name__ == "__main__":
    main()
