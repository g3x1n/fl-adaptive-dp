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
        "optimizer_momentum": 0.0,
        "weight_decay": 0.0,
        "nesterov": False,
        "lr_schedule": "constant",
        "lr_decay_gamma": 1.0,
        "lr_decay_step": 1,
        "min_learning_rate": 0.0,
        "fraction_fit": 1.0,
        "proximal_mu": 0.0,
    },
    "privacy": {
        "dp_mode": "none",
        "epsilon": None,
        "delta": 1e-5,
        "clip_norm": 1.0,
        "noise_multiplier": 0.0,
        "noise_schedule": "none",
        "target_epsilon": None,
        "max_noise_multiplier": 1.2,
        "min_noise_multiplier": 0.2,
        "schedule_metric": "update_norm",
        "schedule_warmup_rounds": 0,
        "adaptive_step": 0.05,
        "plateau_patience": 2,
        "schedule_metric_tolerance": 0.002,
        "schedule_drop_tolerance": 0.01,
        "budget_tolerance_ratio": 0.1,
        "client_aware_clipping": False,
        "client_clipping_metric": "update_norm",
        "client_clipping_beta": 0.5,
        "min_clip_norm": 0.5,
        "max_clip_norm": 1.5,
        "client_aware_noise": False,
        "client_noise_beta": 0.25,
        "reliability_aware_aggregation": False,
        "aggregation_reliability_beta": 0.4,
        "min_reliability_multiplier": 0.6,
        "max_reliability_multiplier": 1.0,
        "accountant": "gaussian",
    },
    "compression": {
        "mode": "none",
        "topk_ratio": 1.0,
        "compress_updates": True,
    },
    "runtime": {
        "device": "auto",
        "num_workers": 0,
        "pin_memory": "auto",
        "persistent_workers": "auto",
        "allow_tf32": True,
        "cudnn_benchmark": True,
        "matmul_precision": "high",
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

    raw = _load_raw_config(config_path, seen_paths=set())

    if not isinstance(raw, dict):
        raise ValueError("Top-level config must be a mapping.")

    config = _deep_merge(DEFAULT_CONFIG, raw)
    config["meta"] = {
        "config_path": str(config_path.resolve()),
    }
    return config


def _load_raw_config(path: Path, seen_paths: set[Path]) -> dict[str, Any]:
    resolved_path = path.resolve()
    if resolved_path in seen_paths:
        raise ValueError(f"Detected recursive config inheritance: {resolved_path}")

    seen_paths.add(resolved_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError("Top-level config must be a mapping.")

    inherited = raw.pop("inherits", None)
    if inherited is None:
        seen_paths.remove(resolved_path)
        return raw

    parent_paths = inherited if isinstance(inherited, list) else [inherited]
    merged_parent: dict[str, Any] = {}
    for parent in parent_paths:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = (path.parent / parent_path).resolve()
        parent_raw = _load_raw_config(parent_path, seen_paths)
        merged_parent = _deep_merge(merged_parent, parent_raw)

    seen_paths.remove(resolved_path)
    return _deep_merge(merged_parent, raw)


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
