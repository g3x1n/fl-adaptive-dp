"""Simple privacy accounting utilities for audit-friendly experiments."""

from __future__ import annotations

import math


def estimate_epsilon_increment(
    sample_rate: float,
    noise_multiplier: float,
    delta: float,
    num_steps: int,
) -> float:
    """Estimate the epsilon increase for one local training round.

    This is a lightweight Gaussian-mechanism-inspired approximation intended
    for comparative experiments, not a formal replacement for a full RDP
    accountant. The goal is monotonic, auditable budget tracking.
    """
    if noise_multiplier <= 0 or num_steps <= 0 or sample_rate <= 0:
        return 0.0

    gaussian_constant = math.sqrt(2.0 * math.log(1.25 / delta))
    return sample_rate * gaussian_constant * math.sqrt(num_steps) / noise_multiplier


class PrivacyAccountant:
    """Track cumulative privacy spending across local rounds."""

    def __init__(self) -> None:
        self.total_steps = 0
        self.epsilon_spent = 0.0

    def step(
        self,
        sample_rate: float,
        noise_multiplier: float,
        delta: float,
        num_steps: int,
    ) -> float:
        increment = estimate_epsilon_increment(
            sample_rate=sample_rate,
            noise_multiplier=noise_multiplier,
            delta=delta,
            num_steps=num_steps,
        )
        self.total_steps += num_steps
        self.epsilon_spent += increment
        return self.epsilon_spent

    def state_dict(self) -> dict[str, float]:
        return {
            "total_steps": self.total_steps,
            "epsilon_spent": self.epsilon_spent,
        }
