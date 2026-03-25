"""Federated optimization algorithms and aggregation strategies."""

from src.optim.fedavg import aggregate_fedavg, aggregate_weighted_updates
from src.optim.fednova import aggregate_fednova, aggregate_fednova_updates
from src.optim.fedprox import compute_proximal_term

__all__ = [
    "aggregate_fedavg",
    "aggregate_fednova",
    "aggregate_fednova_updates",
    "aggregate_weighted_updates",
    "compute_proximal_term",
]
