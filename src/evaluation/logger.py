"""Project logging and metric writing utilities."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir


def setup_logger(name: str, log_file: str | Path) -> logging.Logger:
    """Create a file and console logger with a shared formatter."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    ensure_dir(Path(log_file).parent)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


class MetricsWriter:
    """Append per-round metrics to a CSV file with a stable schema."""

    def __init__(self, output_path: str | Path, fieldnames: list[str]) -> None:
        self.output_path = Path(output_path)
        self.fieldnames = fieldnames
        ensure_dir(self.output_path.parent)

        with self.output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        payload = {key: row.get(key) for key in self.fieldnames}
        with self.output_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(payload)


class JsonlWriter:
    """Append structured JSONL rows for richer experiment artifacts."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        ensure_dir(self.output_path.parent)
        self.output_path.write_text("", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def default_round_metric_fieldnames() -> list[str]:
    """Stable CSV schema for per-round plotting."""
    return [
        "round",
        "learning_rate",
        "train_loss",
        "test_loss",
        "test_accuracy",
        "best_test_accuracy",
        "epsilon_spent",
        "upload_payload_bytes",
        "pre_compression_payload_bytes",
        "selected_clients",
        "algorithm",
        "dp_mode",
        "compression_mode",
        "noise_multiplier",
        "noise_multiplier_min",
        "noise_multiplier_max",
        "noise_multiplier_median",
        "clip_norm",
        "clip_norm_min",
        "clip_norm_max",
        "clip_norm_median",
        "schedule_reason",
        "compression_ratio",
        "nnz_params",
        "avg_update_norm",
        "median_update_norm",
        "max_update_norm",
        "avg_grad_norm",
        "median_grad_norm",
        "max_grad_norm",
        "median_client_loss",
        "max_client_loss",
        "median_client_epsilon",
        "max_client_epsilon",
    ]
