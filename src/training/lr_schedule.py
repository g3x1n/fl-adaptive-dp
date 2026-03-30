"""Learning-rate schedule helpers for federated rounds."""

from __future__ import annotations


def resolve_round_learning_rate(training_config: dict, round_idx: int) -> float:
    """Compute the round-specific learning rate from config."""
    base_lr = float(training_config["learning_rate"])
    schedule = str(training_config.get("lr_schedule", "constant")).lower()
    min_lr = float(training_config.get("min_learning_rate", 0.0))

    if schedule == "constant":
        return max(min_lr, base_lr)

    if schedule == "exp":
        gamma = float(training_config.get("lr_decay_gamma", 1.0))
        lr = base_lr * (gamma ** max(round_idx - 1, 0))
        return max(min_lr, lr)

    if schedule == "step":
        gamma = float(training_config.get("lr_decay_gamma", 1.0))
        step = max(int(training_config.get("lr_decay_step", 1)), 1)
        exponent = max((round_idx - 1) // step, 0)
        lr = base_lr * (gamma ** exponent)
        return max(min_lr, lr)

    raise ValueError(f"Unsupported lr_schedule: {schedule}")
