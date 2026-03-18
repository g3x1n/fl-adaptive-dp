"""Tests for the FedNova aggregation rule."""

from __future__ import annotations

from collections import OrderedDict

import torch

from src.optim import aggregate_fednova


def test_fednova_matches_fedavg_when_client_steps_match() -> None:
    global_state = OrderedDict({"weight": torch.tensor([10.0])})
    client_a = OrderedDict({"weight": torch.tensor([8.0])})
    client_b = OrderedDict({"weight": torch.tensor([6.0])})

    aggregated = aggregate_fednova(
        global_state=global_state,
        client_states=[client_a, client_b],
        client_weights=[1, 1],
        client_steps=[2, 2],
    )

    # With equal step counts, FedNova reduces to the standard weighted average.
    assert torch.allclose(aggregated["weight"], torch.tensor([7.0]))


def test_fednova_respects_step_normalization() -> None:
    global_state = OrderedDict({"weight": torch.tensor([10.0])})
    client_a = OrderedDict({"weight": torch.tensor([8.0])})
    client_b = OrderedDict({"weight": torch.tensor([6.0])})

    aggregated = aggregate_fednova(
        global_state=global_state,
        client_states=[client_a, client_b],
        client_weights=[1, 1],
        client_steps=[1, 3],
    )

    # The second client took more local steps, so FedNova normalizes its update
    # before composing the final global step.
    assert torch.allclose(aggregated["weight"], torch.tensor([6.6666665]))
