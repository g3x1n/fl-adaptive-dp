"""Tests for DP mechanisms, accounting, and scheduling."""

from __future__ import annotations

import torch

from src.models.mnist_cnn import MNISTCNN
from src.privacy import (
    AdaptiveDPScheduler,
    PrivacyAccountant,
    clip_and_add_noise,
    compute_client_adaptive_clip,
    compute_client_reliability_multiplier,
    compute_client_risk_boosted_noise,
)


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
    assert stats["post_clip_grad_norm"] <= 0.50001


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


def test_performance_plateau_scheduler_reduces_noise_after_drop() -> None:
    scheduler = AdaptiveDPScheduler(
        {
            "noise_schedule": "performance_plateau",
            "clip_norm": 1.0,
            "min_noise_multiplier": 0.02,
            "max_noise_multiplier": 0.2,
            "schedule_metric": "test_accuracy",
            "schedule_warmup_rounds": 1,
            "adaptive_step": 0.03,
            "plateau_patience": 1,
            "schedule_metric_tolerance": 0.001,
            "schedule_drop_tolerance": 0.01,
        }
    )

    warmup = scheduler.schedule(round_idx=1, total_rounds=20, previous_metrics={"test_accuracy": 0.0})
    improved = scheduler.schedule(round_idx=2, total_rounds=20, previous_metrics={"test_accuracy": 0.70})
    plateau = scheduler.schedule(round_idx=3, total_rounds=20, previous_metrics={"test_accuracy": 0.701})
    dropped = scheduler.schedule(round_idx=4, total_rounds=20, previous_metrics={"test_accuracy": 0.65})

    assert warmup["noise_multiplier"] == 0.02
    assert improved["noise_multiplier"] == 0.02
    assert plateau["noise_multiplier"] > improved["noise_multiplier"]
    assert dropped["noise_multiplier"] < plateau["noise_multiplier"]


def test_client_adaptive_clip_tightens_large_updates_and_relaxes_small_updates() -> None:
    high = compute_client_adaptive_clip(
        base_clip_norm=1.0,
        client_metric=2.0,
        reference_metric=1.0,
        beta=0.5,
        min_clip_norm=0.5,
        max_clip_norm=1.5,
    )
    low = compute_client_adaptive_clip(
        base_clip_norm=1.0,
        client_metric=0.5,
        reference_metric=1.0,
        beta=0.5,
        min_clip_norm=0.5,
        max_clip_norm=1.5,
    )

    assert high < 1.0
    assert low > 1.0


def test_performance_budget_scheduler_increases_noise_when_budget_runs_hot() -> None:
    scheduler = AdaptiveDPScheduler(
        {
            "noise_schedule": "performance_budget",
            "clip_norm": 1.0,
            "min_noise_multiplier": 0.02,
            "max_noise_multiplier": 0.12,
            "schedule_metric": "test_accuracy",
            "schedule_warmup_rounds": 1,
            "adaptive_step": 0.01,
            "plateau_patience": 2,
            "schedule_metric_tolerance": 0.003,
            "schedule_drop_tolerance": 0.02,
            "target_epsilon": 10.0,
            "budget_tolerance_ratio": 0.1,
        }
    )

    scheduler.schedule(round_idx=1, total_rounds=20, previous_metrics={"test_accuracy": 0.0, "epsilon_spent": 0.0})
    hot = scheduler.schedule(round_idx=2, total_rounds=20, previous_metrics={"test_accuracy": 0.7, "epsilon_spent": 5.0})

    assert hot["noise_multiplier"] > 0.02


def test_client_risk_boosted_noise_increases_for_risky_clients() -> None:
    risky = compute_client_risk_boosted_noise(
        base_noise_multiplier=0.04,
        client_metric=2.0,
        reference_metric=1.0,
        client_num_samples=200,
        reference_num_samples=1000,
        beta=0.25,
        min_noise_multiplier=0.04,
        max_noise_multiplier=0.14,
    )
    stable = compute_client_risk_boosted_noise(
        base_noise_multiplier=0.04,
        client_metric=0.8,
        reference_metric=1.0,
        client_num_samples=1200,
        reference_num_samples=1000,
        beta=0.25,
        min_noise_multiplier=0.04,
        max_noise_multiplier=0.14,
    )

    assert risky > stable
    assert stable == 0.04


def test_client_reliability_multiplier_downweights_noisy_outliers() -> None:
    outlier = compute_client_reliability_multiplier(
        client_metric=2.0,
        reference_metric=1.0,
        client_noise_multiplier=0.14,
        reference_noise_multiplier=0.10,
        beta=0.5,
        min_multiplier=0.6,
        max_multiplier=1.0,
    )
    typical = compute_client_reliability_multiplier(
        client_metric=1.0,
        reference_metric=1.0,
        client_noise_multiplier=0.10,
        reference_noise_multiplier=0.10,
        beta=0.5,
        min_multiplier=0.6,
        max_multiplier=1.0,
    )

    assert outlier < typical
    assert typical == 1.0
