"""Summarize and plot the formal baseline comparison experiment.

This script reads the six run directories of Experiment 1, extracts the most
important metrics used in the thesis, and produces:

- a machine-readable CSV summary
- accuracy / loss curves for IID and Non-IID

The script is intentionally small and explicit so it is easy to audit later.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RUN_SPECS = [
    ("FedAvg", "IID", "exp1_opt_fedavg_mnist_iid"),
    ("FedProx", "IID", "exp1_opt_fedprox_mnist_iid"),
    ("FedNova", "IID", "exp1_opt_fednova_mnist_iid"),
    ("FedAvg", "Non-IID", "exp1_opt_fedavg_mnist_noniid"),
    ("FedProx", "Non-IID", "exp1_opt_fedprox_mnist_noniid"),
    ("FedNova", "Non-IID", "exp1_opt_fednova_mnist_noniid"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Experiment 1 baseline comparison results.")
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Directory that stores experiment output folders.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("docs/实验记录/实验1-正式版"),
        help="Directory to store summary CSV and figures.",
    )
    return parser.parse_args()


def find_run_dir(outputs_root: Path, run_name: str) -> Path:
    candidates = sorted(outputs_root.glob(f"*_{run_name}"))
    if not candidates:
        raise FileNotFoundError(f"Could not find output directory for run: {run_name}")
    return candidates[-1]


def load_metrics(run_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "metrics.csv")
    # Round 0 is the initialization row and should not participate in model metrics.
    frame = frame[frame["test_accuracy"].notna()].copy()
    frame["round"] = frame["round"].astype(int)
    frame["test_accuracy"] = frame["test_accuracy"].astype(float)
    frame["test_loss"] = frame["test_loss"].astype(float)
    frame["train_loss"] = frame["train_loss"].astype(float)
    frame["upload_payload_bytes"] = frame["upload_payload_bytes"].astype(float)
    return frame


def build_summary_rows(outputs_root: Path) -> tuple[list[dict], dict[tuple[str, str], pd.DataFrame], dict[tuple[str, str], Path]]:
    summary_rows: list[dict] = []
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    run_dirs: dict[tuple[str, str], Path] = {}

    for algorithm, scenario, run_name in RUN_SPECS:
        run_dir = find_run_dir(outputs_root, run_name)
        frame = load_metrics(run_dir)
        best_row = frame.loc[frame["test_accuracy"].idxmax()]
        final_row = frame.iloc[-1]

        summary_rows.append(
            {
                "algorithm": algorithm,
                "scenario": scenario,
                "run_name": run_name,
                "run_dir": str(run_dir.resolve()),
                "best_round": int(best_row["round"]),
                "best_test_accuracy": float(best_row["test_accuracy"]),
                "best_test_loss": float(best_row["test_loss"]),
                "final_round": int(final_row["round"]),
                "final_test_accuracy": float(final_row["test_accuracy"]),
                "final_test_loss": float(final_row["test_loss"]),
                "avg_train_loss_last5": float(frame.tail(5)["train_loss"].mean()),
                "total_upload_payload_bytes": float(frame["upload_payload_bytes"].sum()),
            }
        )

        frames[(algorithm, scenario)] = frame
        run_dirs[(algorithm, scenario)] = run_dir

    return summary_rows, frames, run_dirs


def plot_metric(
    artifact_dir: Path,
    frames: dict[tuple[str, str], pd.DataFrame],
    scenario: str,
    metric: str,
    ylabel: str,
    filename: str,
) -> None:
    plt.figure(figsize=(8, 5))
    for algorithm in ["FedAvg", "FedProx", "FedNova"]:
        frame = frames[(algorithm, scenario)]
        plt.plot(frame["round"], frame[metric], marker="o", linewidth=1.8, markersize=3.5, label=algorithm)

    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(f"Experiment 1 {scenario} {ylabel} Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifact_dir / filename, dpi=180)
    plt.close()


def write_summary_csv(artifact_dir: Path, rows: list[dict]) -> Path:
    output_path = artifact_dir / "experiment1_baseline_summary.csv"
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    summary_rows, frames, _ = build_summary_rows(args.outputs_root)
    write_summary_csv(args.artifact_dir, summary_rows)

    plot_metric(args.artifact_dir, frames, "IID", "test_accuracy", "Test Accuracy", "iid_accuracy_curve.png")
    plot_metric(args.artifact_dir, frames, "IID", "test_loss", "Test Loss", "iid_loss_curve.png")
    plot_metric(args.artifact_dir, frames, "Non-IID", "test_accuracy", "Test Accuracy", "noniid_accuracy_curve.png")
    plot_metric(args.artifact_dir, frames, "Non-IID", "test_loss", "Test Loss", "noniid_loss_curve.png")


if __name__ == "__main__":
    main()
