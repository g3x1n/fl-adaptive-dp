"""Project logging and metric writing utilities."""

from __future__ import annotations

import csv
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

