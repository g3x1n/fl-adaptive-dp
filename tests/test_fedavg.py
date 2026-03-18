"""Tests for the minimal FedAvg aggregation logic."""

from __future__ import annotations

from collections import OrderedDict

import torch

from src.optim import aggregate_fedavg


def test_aggregate_fedavg_computes_weighted_average() -> None:
    state_a = OrderedDict({"weight": torch.tensor([1.0, 3.0])})
    state_b = OrderedDict({"weight": torch.tensor([5.0, 7.0])})

    aggregated = aggregate_fedavg([state_a, state_b], [1, 3])

    expected = torch.tensor([4.0, 6.0])
    assert torch.allclose(aggregated["weight"], expected)
