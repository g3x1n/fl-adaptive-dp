"""Tests for membership inference helpers."""

from __future__ import annotations

from src.privacy.mia import _best_threshold_accuracy, _pairwise_auc


def test_pairwise_auc_is_high_when_member_scores_dominate() -> None:
    auc = _pairwise_auc([0.9, 0.8, 0.7], [0.2, 0.3, 0.4])
    assert auc == 1.0


def test_best_threshold_accuracy_beats_random_guess() -> None:
    accuracy = _best_threshold_accuracy([0.9, 0.8, 0.7], [0.4, 0.3, 0.2])
    assert accuracy >= 0.99
