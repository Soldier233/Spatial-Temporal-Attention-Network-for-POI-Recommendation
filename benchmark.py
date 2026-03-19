import argparse
from pathlib import Path

import joblib
import numpy as np
import torch

from layers import hours, resolve_device
from models import Model
from train import evaluate_model, load_checkpoint_records


def load_processed_tensors(dataset, data_dir, device, part):
    data_path = Path(data_dir) / f"{dataset}_data.pkl"
    with data_path.open("rb") as handle:
        file_data = joblib.load(handle)

    if len(file_data) == 11:
        trajs, mat1, mat2s, mat2t, semantic, social, labels, lens, u_max, l_max, _meta = file_data
    else:
        trajs, mat1, mat2s, mat2t, semantic, social, labels, lens, u_max, l_max = file_data

    mat1 = torch.as_tensor(mat1, dtype=torch.float32)
    mat2s = torch.as_tensor(mat2s, dtype=torch.float32, device=device)
    mat2t = torch.as_tensor(mat2t, dtype=torch.float32)
    semantic = torch.as_tensor(semantic, dtype=torch.float32, device=device)
    social = torch.as_tensor(social, dtype=torch.float32, device=device)
    labels = torch.as_tensor(labels, dtype=torch.long)
    lens = torch.as_tensor(lens, dtype=torch.long)

    if part > 0:
        trajs = trajs[:part]
        mat1 = mat1[:part]
        mat2t = mat2t[:part]
        labels = labels[:part]
        lens = lens[:part]

    return (trajs, mat1, mat2s, mat2t, semantic, social, labels, lens, u_max, l_max)


def build_model_from_checkpoint(checkpoint, mat1, u_max, l_max, device):
    ex = (mat1[:, :, :, 0].max(), mat1[:, :, :, 0].min(), mat1[:, :, :, 1].max(), mat1[:, :, :, 1].min())
    state_dict = checkpoint["state_dict"]
    embed_dim = int(state_dict["MultiEmbed.emb_t.weight"].shape[1])
    model = Model(t_dim=hours + 1, l_dim=l_max + 1, u_dim=u_max + 1, embed_dim=embed_dim, ex=ex, dropout=0)
    model.load_state_dict(state_dict)
    return model.to(device)

def evaluate_checkpoint(checkpoint_path, tensors, device, batch_size):
    checkpoint, _records = load_checkpoint_records(checkpoint_path, device)
    model = build_model_from_checkpoint(checkpoint, tensors[1], tensors[8], tensors[9], device)
    metrics = evaluate_model(model, tensors[:8], device, batch_size=batch_size)
    metrics["checkpoint"] = str(checkpoint_path)
    return metrics


def find_checkpoints(dataset, checkpoint, checkpoint_glob):
    if checkpoint_glob:
        paths = sorted(Path().glob(checkpoint_glob))
    else:
        target = Path(checkpoint or f"best_stan_{dataset}.pth")
        paths = [target]
    existing = [path for path in paths if path.exists()]
    if not existing:
        raise FileNotFoundError("No checkpoints matched the provided checkpoint path or glob.")
    return existing


def summarize_runs(metrics_list):
    valid5 = np.array([item["valid"][1] for item in metrics_list], dtype=np.float64)
    valid10 = np.array([item["valid"][2] for item in metrics_list], dtype=np.float64)
    test5 = np.array([item["test"][1] for item in metrics_list], dtype=np.float64)
    test10 = np.array([item["test"][2] for item in metrics_list], dtype=np.float64)
    return {
        "valid@5": valid5,
        "valid@10": valid10,
        "test@5": test5,
        "test@10": test10,
    }


def paired_t_test(a, b):
    try:
        from scipy import stats
    except ImportError as exc:
        raise ImportError("scipy is required for the paired t-test. Install it with `pip install scipy`.") from exc

    t_stat, p_value = stats.ttest_rel(a, b)
    return float(t_stat), float(p_value)


def print_metric_line(name, values):
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    print(f"{name}: mean={mean:.4f}, std={std:.4f}, runs={len(values)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark STAN best checkpoints on recall@5 and recall@10.")
    parser.add_argument("--dataset", default="NYC")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--part", type=int, default=-1, help="Number of users to evaluate. Use -1 for all users.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--checkpoint", default=None, help="Single checkpoint path. Defaults to best_stan_<dataset>.pth.")
    parser.add_argument("--checkpoint-glob", default=None, help="Glob for multiple STAN checkpoints, e.g. 'runs/*/best*.pth'.")
    parser.add_argument("--baseline-glob", default=None, help="Glob for baseline checkpoints for paired t-test.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    tensors = load_processed_tensors(args.dataset, args.data_dir, device, args.part)

    stan_paths = find_checkpoints(args.dataset, args.checkpoint, args.checkpoint_glob)
    stan_metrics = [evaluate_checkpoint(path, tensors, device, args.batch_size) for path in stan_paths]
    stan_summary = summarize_runs(stan_metrics)

    print("STAN benchmark")
    for item in stan_metrics:
        print(
            f"checkpoint:{item['checkpoint']}, "
            f"valid_recall@5:{item['valid'][1]:.4f}, valid_recall@10:{item['valid'][2]:.4f}, "
            f"test_recall@5:{item['test'][1]:.4f}, test_recall@10:{item['test'][2]:.4f}"
        )

    print_metric_line("valid_recall@5", stan_summary["valid@5"])
    print_metric_line("valid_recall@10", stan_summary["valid@10"])
    print_metric_line("test_recall@5", stan_summary["test@5"])
    print_metric_line("test_recall@10", stan_summary["test@10"])

    if args.baseline_glob:
        baseline_paths = find_checkpoints(args.dataset, None, args.baseline_glob)
        if len(baseline_paths) != len(stan_paths):
            raise ValueError("Paired t-test requires the same number of STAN and baseline checkpoints.")

        baseline_metrics = [evaluate_checkpoint(path, tensors, device, args.batch_size) for path in baseline_paths]
        baseline_summary = summarize_runs(baseline_metrics)

        print("Baseline benchmark")
        for item in baseline_metrics:
            print(
                f"checkpoint:{item['checkpoint']}, "
                f"valid_recall@5:{item['valid'][1]:.4f}, valid_recall@10:{item['valid'][2]:.4f}, "
                f"test_recall@5:{item['test'][1]:.4f}, test_recall@10:{item['test'][2]:.4f}"
            )

        print_metric_line("baseline_valid_recall@5", baseline_summary["valid@5"])
        print_metric_line("baseline_valid_recall@10", baseline_summary["valid@10"])
        print_metric_line("baseline_test_recall@5", baseline_summary["test@5"])
        print_metric_line("baseline_test_recall@10", baseline_summary["test@10"])

        print("Paired t-test (paper style: 10-run average, reject H0 if p < 0.01)")
        for metric_name in ("valid@5", "valid@10", "test@5", "test@10"):
            t_stat, p_value = paired_t_test(stan_summary[metric_name], baseline_summary[metric_name])
            decision = "reject H0" if p_value < 0.01 else "fail to reject H0"
            print(f"{metric_name}: t={t_stat:.4f}, p={p_value:.6f}, decision={decision}")
    elif len(stan_paths) < 10:
        print(
            "Note: the paper reports averaged results over 10 independent runs and uses a paired t-test at p=0.01. "
            "For that workflow, pass 10 STAN checkpoints with --checkpoint-glob and 10 baseline checkpoints with --baseline-glob."
        )


if __name__ == "__main__":
    main()
