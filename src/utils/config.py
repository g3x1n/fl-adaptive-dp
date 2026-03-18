"""Configuration loading and normalization helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "debug_experiment",
        "seed": 42,
        "output_root": "outputs",
    },
    "dataset": {
        "name": "mnist",
        "root": "data",
        "partition_mode": "dirichlet",
        "num_clients": 10,
        "dirichlet_alpha": 0.5,
        "download": False,
        "num_classes": 10,
        "max_train_samples": None,
        "max_test_samples": None,
    },
    "model": {
        "name": "mnist_cnn",
    },
    "training": {
        "algorithm": "fedavg",
        "rounds": 10,
        "local_epochs": 1,
        "batch_size": 32,
        "eval_batch_size": 128,
        "learning_rate": 0.01,
        "fraction_fit": 1.0,
        "proximal_mu": 0.0,
    },
    "privacy": {
        "dp_mode": "none",
        "epsilon": None,
        "noise_schedule": "none",
    },
    "compression": {
        "mode": "none",
        "topk_ratio": 1.0,
    },
    "runtime": {
        "device": "auto",
        "num_workers": 0,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and merge it with project defaults."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError("Top-level config must be a mapping.")

    config = _deep_merge(DEFAULT_CONFIG, raw)
    config["meta"] = {
        "config_path": str(config_path.resolve()),
    }
    return config


def flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested config values for logging or summaries."""
    flattened: dict[str, Any] = {}
    for key, value in config.items():
        next_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_config(value, next_key))
        else:
            flattened[next_key] = value
    return flattened
