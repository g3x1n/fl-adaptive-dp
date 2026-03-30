"""Helpers for normalizing and validating privacy-related config values."""

from __future__ import annotations

from typing import Any


def resolve_privacy_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize privacy config into a runtime-friendly dictionary.

    The project already merges user YAML with defaults, so this helper mainly
    makes types explicit and ensures downstream code can rely on stable keys.
    """
    privacy = dict(config.get("privacy", {}))
    mode = str(privacy.get("dp_mode", "none")).lower()

    resolved = {
        "dp_mode": mode,
        "epsilon": privacy.get("epsilon"),
        "delta": float(privacy.get("delta", 1e-5)),
        "clip_norm": float(privacy.get("clip_norm", 1.0)),
        "noise_multiplier": float(privacy.get("noise_multiplier", 0.0)),
        "noise_schedule": str(privacy.get("noise_schedule", "none")).lower(),
        "target_epsilon": privacy.get("target_epsilon"),
        "max_noise_multiplier": float(privacy.get("max_noise_multiplier", privacy.get("noise_multiplier", 0.0) or 0.0)),
        "min_noise_multiplier": float(privacy.get("min_noise_multiplier", privacy.get("noise_multiplier", 0.0) or 0.0)),
        "schedule_metric": str(privacy.get("schedule_metric", "update_norm")).lower(),
        "schedule_warmup_rounds": int(privacy.get("schedule_warmup_rounds", 0)),
        "adaptive_step": float(privacy.get("adaptive_step", 0.05)),
        "plateau_patience": int(privacy.get("plateau_patience", 2)),
        "schedule_metric_tolerance": float(privacy.get("schedule_metric_tolerance", 0.002)),
        "schedule_drop_tolerance": float(privacy.get("schedule_drop_tolerance", 0.01)),
        "budget_tolerance_ratio": float(privacy.get("budget_tolerance_ratio", 0.1)),
        "client_aware_clipping": bool(privacy.get("client_aware_clipping", False)),
        "client_clipping_metric": str(privacy.get("client_clipping_metric", "update_norm")).lower(),
        "client_clipping_beta": float(privacy.get("client_clipping_beta", 0.5)),
        "min_clip_norm": float(privacy.get("min_clip_norm", privacy.get("clip_norm", 1.0))),
        "max_clip_norm": float(privacy.get("max_clip_norm", privacy.get("clip_norm", 1.0))),
        "client_aware_noise": bool(privacy.get("client_aware_noise", False)),
        "client_noise_beta": float(privacy.get("client_noise_beta", 0.25)),
        "reliability_aware_aggregation": bool(privacy.get("reliability_aware_aggregation", False)),
        "aggregation_reliability_beta": float(privacy.get("aggregation_reliability_beta", 0.4)),
        "min_reliability_multiplier": float(privacy.get("min_reliability_multiplier", 0.6)),
        "max_reliability_multiplier": float(privacy.get("max_reliability_multiplier", 1.0)),
        "accountant": str(privacy.get("accountant", "gaussian")).lower(),
    }

    if mode == "none":
        resolved["noise_multiplier"] = 0.0

    if resolved["clip_norm"] <= 0:
        raise ValueError("privacy.clip_norm must be positive.")

    if resolved["delta"] <= 0 or resolved["delta"] >= 1:
        raise ValueError("privacy.delta must be in the open interval (0, 1).")

    if resolved["min_noise_multiplier"] < 0 or resolved["max_noise_multiplier"] < 0:
        raise ValueError("Noise multipliers must be non-negative.")

    if resolved["min_noise_multiplier"] > resolved["max_noise_multiplier"]:
        raise ValueError("privacy.min_noise_multiplier cannot exceed privacy.max_noise_multiplier.")

    if resolved["adaptive_step"] < 0:
        raise ValueError("privacy.adaptive_step must be non-negative.")

    if resolved["plateau_patience"] < 1:
        raise ValueError("privacy.plateau_patience must be at least 1.")

    if resolved["client_clipping_beta"] < 0:
        raise ValueError("privacy.client_clipping_beta must be non-negative.")

    if resolved["client_noise_beta"] < 0:
        raise ValueError("privacy.client_noise_beta must be non-negative.")

    if resolved["aggregation_reliability_beta"] < 0:
        raise ValueError("privacy.aggregation_reliability_beta must be non-negative.")

    if resolved["budget_tolerance_ratio"] < 0:
        raise ValueError("privacy.budget_tolerance_ratio must be non-negative.")

    if resolved["min_clip_norm"] <= 0 or resolved["max_clip_norm"] <= 0:
        raise ValueError("privacy min/max clip norms must be positive.")

    if resolved["min_clip_norm"] > resolved["max_clip_norm"]:
        raise ValueError("privacy.min_clip_norm cannot exceed privacy.max_clip_norm.")

    if resolved["min_reliability_multiplier"] <= 0 or resolved["max_reliability_multiplier"] <= 0:
        raise ValueError("privacy reliability multipliers must be positive.")

    if resolved["min_reliability_multiplier"] > resolved["max_reliability_multiplier"]:
        raise ValueError(
            "privacy.min_reliability_multiplier cannot exceed privacy.max_reliability_multiplier."
        )

    return resolved
