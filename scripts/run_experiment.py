"""Minimal experiment entrypoint for config and logging verification."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.logger import MetricsWriter, setup_logger
from src.fl import run_federated_experiment
from src.utils.config import flatten_config, load_config
from src.utils.io import dump_json, dump_yaml, ensure_dir, timestamp
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single experiment scaffold.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML experiment config.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional custom run directory suffix.",
    )
    return parser.parse_args()


def build_run_dir(config: dict, run_name: str | None) -> Path:
    base_name = run_name or config["experiment"]["name"]
    output_root = Path(config["experiment"]["output_root"])
    run_dir = output_root / f"{timestamp()}_{base_name}"
    ensure_dir(run_dir)
    ensure_dir(run_dir / "figures")
    return run_dir


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_global_seed(config["experiment"]["seed"])

    run_dir = build_run_dir(config, args.run_name)
    logger = setup_logger("experiment", run_dir / "train.log")

    dump_yaml(run_dir / "config.yaml", config)
    dump_json(
        run_dir / "summary.json",
        {
            "status": "initialized",
            "run_dir": str(run_dir.resolve()),
            "algorithm": config["training"]["algorithm"],
            "dataset": config["dataset"]["name"],
        },
    )

    metrics_writer = MetricsWriter(
        run_dir / "metrics.csv",
        fieldnames=[
            "round",
            "train_loss",
            "test_loss",
            "test_accuracy",
            "epsilon_spent",
            "upload_payload_bytes",
            "selected_clients",
            "algorithm",
            "dp_mode",
            "compression_mode",
        ],
    )

    metrics_writer.write(
        {
            "round": 0,
            "train_loss": None,
            "test_loss": None,
            "test_accuracy": None,
            "epsilon_spent": None,
            "upload_payload_bytes": None,
            "selected_clients": config["dataset"]["num_clients"],
            "algorithm": config["training"]["algorithm"],
            "dp_mode": config["privacy"]["dp_mode"],
            "compression_mode": config["compression"]["mode"],
        }
    )

    logger.info("Experiment scaffold initialized.")
    logger.info("Run directory: %s", run_dir.resolve())
    logger.info("Flattened config: %s", flatten_config(config))

    if config["training"]["algorithm"].lower() not in {"fedavg", "fedprox", "fednova"}:
        raise ValueError("The current minimal training loop only supports FedAvg, FedProx and FedNova.")

    results = run_federated_experiment(
        config=config,
        logger=logger,
        metrics_writer=metrics_writer,
    )
    dump_json(
        run_dir / "summary.json",
        {
            "status": "completed",
            "run_dir": str(run_dir.resolve()),
            "algorithm": config["training"]["algorithm"],
            "dataset": config["dataset"]["name"],
            "results": results,
        },
    )
    logger.info("Training finished with results: %s", results)


if __name__ == "__main__":
    main()
