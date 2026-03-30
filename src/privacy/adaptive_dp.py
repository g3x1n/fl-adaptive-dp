"""Adaptive DP scheduling strategies with per-round audit reasons."""

from __future__ import annotations

from typing import Any


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


class AdaptiveDPScheduler:
    """Schedule per-round DP strength using explicit, auditable rules."""

    def __init__(self, privacy_config: dict[str, Any]) -> None:
        self.privacy_config = privacy_config
        self.best_metric = float("-inf")
        self.plateau_count = 0
        self.current_noise = float(privacy_config["min_noise_multiplier"])

    def schedule(
        self,
        round_idx: int,
        total_rounds: int,
        previous_metrics: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        schedule_name = self.privacy_config["noise_schedule"]
        if schedule_name == "performance_budget":
            return self._performance_budget(round_idx, total_rounds, previous_metrics or {})
        if schedule_name == "performance_plateau":
            return self._performance_plateau(round_idx, total_rounds, previous_metrics or {})
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

    def _performance_plateau(
        self,
        round_idx: int,
        total_rounds: int,
        previous_metrics: dict[str, float],
    ) -> dict[str, Any]:
        min_noise = float(self.privacy_config["min_noise_multiplier"])
        max_noise = float(self.privacy_config["max_noise_multiplier"])
        clip_norm = float(self.privacy_config["clip_norm"])
        metric_name = str(self.privacy_config.get("schedule_metric", "test_accuracy")).lower()
        metric_value = float(previous_metrics.get(metric_name, previous_metrics.get("test_accuracy", 0.0)) or 0.0)
        warmup_rounds = int(self.privacy_config["schedule_warmup_rounds"])
        step = float(self.privacy_config.get("adaptive_step", 0.05))
        patience = int(self.privacy_config.get("plateau_patience", 2))
        improvement_tol = float(self.privacy_config.get("schedule_metric_tolerance", 0.002))
        drop_tol = float(self.privacy_config.get("schedule_drop_tolerance", 0.01))

        if round_idx <= warmup_rounds:
            self.current_noise = min_noise
            self.plateau_count = 0
            reason = f"performance-plateau warmup until round {warmup_rounds}"
        else:
            if metric_value > self.best_metric + improvement_tol:
                self.best_metric = metric_value
                self.plateau_count = 0
                self.current_noise = max(min_noise, self.current_noise - step)
                reason = (
                    f"metric improved ({metric_name}={metric_value:.4f}), "
                    f"keep low noise"
                )
            elif self.best_metric - metric_value > drop_tol:
                self.plateau_count = 0
                self.current_noise = max(min_noise, self.current_noise - step)
                reason = (
                    f"metric degraded ({metric_name}={metric_value:.4f}, best={self.best_metric:.4f}), "
                    f"reduce noise to recover stability"
                )
            else:
                self.plateau_count += 1
                if self.plateau_count >= patience:
                    self.current_noise = min(max_noise, self.current_noise + step)
                    self.plateau_count = 0
                    reason = (
                        f"metric plateaued ({metric_name}={metric_value:.4f}), "
                        f"increase noise by step={step:.4f}"
                    )
                else:
                    reason = (
                        f"short plateau ({metric_name}={metric_value:.4f}), "
                        f"hold noise for patience={patience}"
                    )

        return {
            "dp_mode": "adaptive",
            "clip_norm": clip_norm,
            "noise_multiplier": float(_clamp(self.current_noise, min_noise, max_noise)),
            "schedule_reason": reason,
        }

    def _performance_budget(
        self,
        round_idx: int,
        total_rounds: int,
        previous_metrics: dict[str, float],
    ) -> dict[str, Any]:
        plan = self._performance_plateau(round_idx, total_rounds, previous_metrics)

        min_noise = float(self.privacy_config["min_noise_multiplier"])
        max_noise = float(self.privacy_config["max_noise_multiplier"])
        step = float(self.privacy_config.get("adaptive_step", 0.05))
        target_epsilon = self.privacy_config.get("target_epsilon")
        epsilon_spent = float(previous_metrics.get("epsilon_spent", 0.0) or 0.0)
        tolerance_ratio = float(self.privacy_config.get("budget_tolerance_ratio", 0.1))

        if target_epsilon is None or round_idx <= self.privacy_config["schedule_warmup_rounds"]:
            return plan

        expected_spent = float(target_epsilon) * max(round_idx - 1, 1) / max(total_rounds, 1)
        upper_bound = expected_spent * (1.0 + tolerance_ratio)
        lower_bound = expected_spent * max(0.0, 1.0 - tolerance_ratio)

        if epsilon_spent > upper_bound:
            self.current_noise = min(max_noise, self.current_noise + step)
            plan["noise_multiplier"] = float(_clamp(self.current_noise, min_noise, max_noise))
            plan["schedule_reason"] = (
                f"{plan['schedule_reason']}; budget pressure epsilon={epsilon_spent:.4f} "
                f"> expected_upper={upper_bound:.4f}, increase noise"
            )
        elif epsilon_spent < lower_bound and previous_metrics.get("test_accuracy", 0.0) > 0:
            self.current_noise = max(min_noise, self.current_noise - step)
            plan["noise_multiplier"] = float(_clamp(self.current_noise, min_noise, max_noise))
            plan["schedule_reason"] = (
                f"{plan['schedule_reason']}; budget slack epsilon={epsilon_spent:.4f} "
                f"< expected_lower={lower_bound:.4f}, allow lower noise"
            )

        return plan
