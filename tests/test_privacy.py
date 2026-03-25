"""Tests for DP mechanisms, accounting, and scheduling."""

from __future__ import annotations

import torch

from src.models.mnist_cnn import MNISTCNN
from src.privacy import AdaptiveDPScheduler, PrivacyAccountant, clip_and_add_noise


def test_clip_and_add_noise_bounds_the_gradient_norm() -> None:
    model = MNISTCNN()
    inputs = torch.randn(4, 1, 28, 28)
    targets = torch.tensor([0, 1, 2, 3])
    criterion = torch.nn.CrossEntropyLoss()

    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()

    stats = clip_and_add_noise(model, clip_norm=0.5, noise_multiplier=0.0)

    assert stats["pre_clip_grad_norm"] >= stats["post_clip_grad_norm"]
    assert stats["post_clip_grad_norm"] <= 0.500001


def test_privacy_accountant_is_monotonic() -> None:
    accountant = PrivacyAccountant()
    epsilon_a = accountant.step(sample_rate=0.1, noise_multiplier=1.0, delta=1e-5, num_steps=10)
    epsilon_b = accountant.step(sample_rate=0.1, noise_multiplier=1.0, delta=1e-5, num_steps=10)

    assert epsilon_a > 0
    assert epsilon_b > epsilon_a


def test_round_based_scheduler_changes_noise_across_rounds() -> None:
    scheduler = AdaptiveDPScheduler(
        {
            "noise_schedule": "round_based",
            "clip_norm": 1.0,
            "min_noise_multiplier": 0.2,
            "max_noise_multiplier": 1.0,
            "schedule_warmup_rounds": 0,
        }
    )

    early = scheduler.schedule(round_idx=1, total_rounds=10)
    late = scheduler.schedule(round_idx=10, total_rounds=10)

    assert early["noise_multiplier"] < late["noise_multiplier"]


def test_metric_based_scheduler_responds_to_update_norm() -> None:
    scheduler = AdaptiveDPScheduler(
        {
            "noise_schedule": "metric_based",
            "clip_norm": 1.0,
            "min_noise_multiplier": 0.2,
            "max_noise_multiplier": 1.0,
            "schedule_metric": "update_norm",
            "schedule_warmup_rounds": 0,
        }
    )

    low = scheduler.schedule(round_idx=3, total_rounds=10, previous_metrics={"update_norm": 0.1})
    high = scheduler.schedule(round_idx=3, total_rounds=10, previous_metrics={"update_norm": 2.0})

    assert low["noise_multiplier"] < high["noise_multiplier"]
