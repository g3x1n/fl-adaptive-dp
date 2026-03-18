"""Federated optimization algorithms and aggregation strategies."""

from src.optim.fedavg import aggregate_fedavg
from src.optim.fednova import aggregate_fednova
from src.optim.fedprox import compute_proximal_term

__all__ = ["aggregate_fedavg", "aggregate_fednova", "compute_proximal_term"]
