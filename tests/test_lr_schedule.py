"""Tests for round-level learning-rate schedules."""

from __future__ import annotations

from src.training import resolve_round_learning_rate


def test_constant_schedule_returns_base_lr() -> None:
    lr = resolve_round_learning_rate(
        {
            "learning_rate": 0.05,
            "lr_schedule": "constant",
            "min_learning_rate": 0.0,
        },
        round_idx=10,
    )

    assert lr == 0.05


def test_exponential_schedule_decays_learning_rate() -> None:
    lr = resolve_round_learning_rate(
        {
            "learning_rate": 0.1,
            "lr_schedule": "exp",
            "lr_decay_gamma": 0.9,
            "min_learning_rate": 0.0,
        },
        round_idx=3,
    )

    assert abs(lr - 0.081) < 1e-9


def test_step_schedule_respects_floor() -> None:
    lr = resolve_round_learning_rate(
        {
            "learning_rate": 0.1,
            "lr_schedule": "step",
            "lr_decay_gamma": 0.1,
            "lr_decay_step": 2,
            "min_learning_rate": 0.02,
        },
        round_idx=7,
    )

    assert lr == 0.02
