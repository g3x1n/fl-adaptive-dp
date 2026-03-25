"""Adaptive DP scheduling strategies with per-round audit reasons."""

from __future__ import annotations

from typing import Any


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


class AdaptiveDPScheduler:
    """Schedule per-round DP strength using explicit, auditable rules."""

    def __init__(self, privacy_config: dict[str, Any]) -> None:
        self.privacy_config = privacy_config

    def schedule(
        self,
        round_idx: int,
        total_rounds: int,
        previous_metrics: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        schedule_name = self.privacy_config["noise_schedule"]
        if schedule_name == "metric_based":
            return self._metric_based(round_idx, total_rounds, previous_metrics or {})
        return self._round_based(round_idx, total_rounds)

    def _round_based(self, round_idx: int, total_rounds: int) -> dict[str, Any]:
        warmup_rounds = self.privacy_config["schedule_warmup_rounds"]
        min_noise = self.privacy_config["min_noise_multiplier"]
        max_noise = self.privacy_config["max_noise_multiplier"]

        if round_idx <= warmup_rounds:
            noise = min_noise
            reason = f"warmup rounds <= {warmup_rounds}, keep minimum noise"
        else:
            effective_rounds = max(total_rounds - warmup_rounds, 1)
            progress = (round_idx - warmup_rounds - 1) / max(effective_rounds - 1, 1)
            noise = min_noise + progress * (max_noise - min_noise)
            reason = f"round-based schedule with progress={progress:.3f}"

        return {
            "dp_mode": "adaptive",
            "clip_norm": float(self.privacy_config["clip_norm"]),
            "noise_multiplier": float(_clamp(noise, min_noise, max_noise)),
            "schedule_reason": reason,
        }

    def _metric_based(
        self,
        round_idx: int,
        total_rounds: int,
        previous_metrics: dict[str, float],
    ) -> dict[str, Any]:
        min_noise = self.privacy_config["min_noise_multiplier"]
        max_noise = self.privacy_config["max_noise_multiplier"]
        schedule_metric = self.privacy_config["schedule_metric"]

        metric_value = float(previous_metrics.get(schedule_metric, previous_metrics.get("update_norm", 0.0)) or 0.0)
        clip_norm = float(self.privacy_config["clip_norm"])

        # A larger update norm means the round revealed stronger gradient
        # information, so we respond with stronger noise in the next round.
        scaled = metric_value / max(clip_norm, 1e-8)
        noise = min_noise + (max_noise - min_noise) * min(scaled, 1.0)

        if round_idx <= self.privacy_config["schedule_warmup_rounds"]:
            noise = min_noise
            reason = f"metric-based warmup, defer adaptation until round {self.privacy_config['schedule_warmup_rounds'] + 1}"
        else:
            reason = f"metric-based schedule using {schedule_metric}={metric_value:.4f}"

        return {
            "dp_mode": "adaptive",
            "clip_norm": clip_norm,
            "noise_multiplier": float(_clamp(noise, min_noise, max_noise)),
            "schedule_reason": reason,
        }
