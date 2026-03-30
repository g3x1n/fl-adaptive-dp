"""Tests for white-box gradient inversion helpers."""

from __future__ import annotations

import torch

from src.models.mnist_cnn import MNISTCNN
from src.privacy.whitebox_attack import (
    collect_observed_gradients,
    infer_label_from_gradients,
)


def test_idlg_recovers_single_sample_label_without_dp() -> None:
    model = MNISTCNN()
    inputs = torch.randn(1, 1, 28, 28)
    targets = torch.tensor([3])

    observed_gradients = collect_observed_gradients(
        model=model,
        inputs=inputs,
        targets=targets,
        dp_mode="none",
        clip_norm=1.0,
        noise_multiplier=0.0,
    )

    recovered_label = infer_label_from_gradients(observed_gradients)
    assert recovered_label == 3
