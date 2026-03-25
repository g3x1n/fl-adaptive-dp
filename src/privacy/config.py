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

    return resolved
