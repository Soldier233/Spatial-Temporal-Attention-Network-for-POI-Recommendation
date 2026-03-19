import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from benchmark import (
    evaluate_checkpoint,
    find_checkpoints,
    load_processed_tensors,
    paired_t_test,
    summarize_runs,
)
from layers import resolve_device


def parse_seed_list(seed_text, repeats):
    if seed_text:
        return [int(item.strip()) for item in seed_text.split(",") if item.strip()]
    return list(range(repeats))


def run_training(args, checkpoint_path, seed):
    cmd = [
        sys.executable,
        "train.py",
        "--dataset",
        args.dataset,
        "--data-dir",
        args.data_dir,
        "--device",
        args.device,
        "--part",
        str(args.part),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.train_batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--num-neg",
        str(args.num_neg),
        "--embed-dim",
        str(args.embed_dim),
        "--seed",
        str(seed),
        "--checkpoint",
        str(checkpoint_path),
    ]

    if args.resume:
        cmd.append("--resume")

    subprocess.run(cmd, check=True)


def write_run_csv(csv_path, rows):
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed",
                "checkpoint",
                "valid_recall@5",
                "valid_recall@10",
                "test_recall@5",
                "test_recall@10",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def metric_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "runs": int(len(values)),
    }


def print_summary(title, summary):
    print(title)
    for metric_name in ("valid@5", "valid@10", "test@5", "test@10"):
        stats = metric_stats(summary[metric_name])
        print(
            f"{metric_name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, runs={stats['runs']}"
        )


def main():
    parser = argparse.ArgumentParser(description="One-click multi-seed STAN training, benchmark, and paper-style summary.")
    parser.add_argument("--dataset", default="NYC")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--part", type=int, default=100, help="Number of users to train/evaluate. Use -1 for all users.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--benchmark-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--num-neg", type=int, default=10)
    parser.add_argument("--embed-dim", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seeds", default=None, help="Comma-separated seed list. Defaults to 0..repeats-1.")
    parser.add_argument("--output-dir", default=None, help="Directory to store checkpoints and benchmark summaries.")
    parser.add_argument("--baseline-glob", default=None, help="Optional baseline checkpoints for paired t-test.")
    parser.add_argument("--force-train", action="store_true", help="Retrain even if the checkpoint already exists.")
    parser.add_argument("--resume", action="store_true", help="Resume each seed from its checkpoint if it exists.")
    args = parser.parse_args()

    seed_list = parse_seed_list(args.seeds, args.repeats)
    output_dir = Path(args.output_dir or f"./paper_runs/{args.dataset}")
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    tensors = load_processed_tensors(args.dataset, args.data_dir, device, args.part)

    run_rows = []
    metrics_list = []
    checkpoint_paths = []

    for seed in seed_list:
        checkpoint_path = checkpoint_dir / f"best_stan_{args.dataset}_seed{seed}.pth"
        checkpoint_paths.append(checkpoint_path)
        if args.force_train or not checkpoint_path.exists():
            print(f"training seed={seed} -> {checkpoint_path}")
            run_training(args, checkpoint_path, seed)
        else:
            print(f"skip existing checkpoint for seed={seed}: {checkpoint_path}")

        metrics = evaluate_checkpoint(checkpoint_path, tensors, device, args.benchmark_batch_size)
        metrics_list.append(metrics)
        run_rows.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                "valid_recall@5": f"{metrics['valid'][1]:.6f}",
                "valid_recall@10": f"{metrics['valid'][2]:.6f}",
                "test_recall@5": f"{metrics['test'][1]:.6f}",
                "test_recall@10": f"{metrics['test'][2]:.6f}",
            }
        )
        print(
            f"seed={seed}, valid_recall@5={metrics['valid'][1]:.4f}, valid_recall@10={metrics['valid'][2]:.4f}, "
            f"test_recall@5={metrics['test'][1]:.4f}, test_recall@10={metrics['test'][2]:.4f}"
        )

    summary = summarize_runs(metrics_list)
    print_summary("STAN paper-style benchmark", summary)

    write_run_csv(output_dir / "stan_runs.csv", run_rows)

    report = {
        "dataset": args.dataset,
        "device": args.device,
        "part": args.part,
        "epochs": args.epochs,
        "seeds": seed_list,
        "checkpoints": [str(path) for path in checkpoint_paths],
        "stan": {metric_name: metric_stats(values) for metric_name, values in summary.items()},
    }

    if args.baseline_glob:
        baseline_paths = find_checkpoints(args.dataset, None, args.baseline_glob)
        if len(baseline_paths) != len(metrics_list):
            raise ValueError("Paired t-test requires the same number of STAN and baseline checkpoints.")

        baseline_metrics = [
            evaluate_checkpoint(path, tensors, device, args.benchmark_batch_size) for path in baseline_paths
        ]
        baseline_summary = summarize_runs(baseline_metrics)
        print_summary("Baseline paper-style benchmark", baseline_summary)

        t_test_report = {}
        print("Paired t-test at p=0.01")
        for metric_name in ("valid@5", "valid@10", "test@5", "test@10"):
            t_stat, p_value = paired_t_test(summary[metric_name], baseline_summary[metric_name])
            decision = "reject H0" if p_value < 0.01 else "fail to reject H0"
            t_test_report[metric_name] = {
                "t_stat": t_stat,
                "p_value": p_value,
                "decision": decision,
            }
            print(f"{metric_name}: t={t_stat:.4f}, p={p_value:.6f}, decision={decision}")

        report["baseline"] = {metric_name: metric_stats(values) for metric_name, values in baseline_summary.items()}
        report["t_test"] = t_test_report

    with (output_dir / "summary.json").open("w") as handle:
        json.dump(report, handle, indent=2)

    print(f"saved run table to {output_dir / 'stan_runs.csv'}")
    print(f"saved summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
