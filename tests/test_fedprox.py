"""Tests for the FedProx proximal term."""

from __future__ import annotations

import torch

from src.models.mnist_cnn import MNISTCNN
from src.optim import compute_proximal_term


def test_proximal_term_is_zero_for_identical_models() -> None:
    model_a = MNISTCNN()
    model_b = MNISTCNN()
    model_b.load_state_dict(model_a.state_dict())

    penalty = compute_proximal_term(model_a, model_b, proximal_mu=0.1)
    assert torch.isclose(penalty, torch.tensor(0.0))


def test_proximal_term_is_positive_for_different_models() -> None:
    model_a = MNISTCNN()
    model_b = MNISTCNN()

    with torch.no_grad():
        next(model_a.parameters()).add_(1.0)

    penalty = compute_proximal_term(model_a, model_b, proximal_mu=0.1)
    assert penalty.item() > 0
