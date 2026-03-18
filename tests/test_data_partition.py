"""Tests for dataset partition helpers."""

from __future__ import annotations

from src.data.partition import partition_dirichlet, partition_iid
from src.data.statistics import summarize_client_distributions, summarize_partition_overview


def test_partition_iid_preserves_all_samples() -> None:
    targets = [index % 10 for index in range(100)]
    partition_map = partition_iid(targets=targets, num_clients=5, seed=42)

    merged = []
    for sample_indices in partition_map.values():
        merged.extend(sample_indices)

    assert len(merged) == len(targets)
    assert sorted(merged) == list(range(len(targets)))


def test_partition_dirichlet_preserves_all_samples() -> None:
    targets = [index % 10 for index in range(200)]
    partition_map = partition_dirichlet(
        targets=targets,
        num_clients=10,
        alpha=0.5,
        seed=7,
    )

    merged = []
    for sample_indices in partition_map.values():
        merged.extend(sample_indices)

    assert len(merged) == len(targets)
    assert sorted(merged) == list(range(len(targets)))


def test_partition_summary_matches_total_samples() -> None:
    targets = [index % 4 for index in range(80)]
    partition_map = partition_iid(targets=targets, num_clients=4, seed=3)

    summaries = summarize_client_distributions(
        targets=targets,
        partition_map=partition_map,
        num_classes=4,
    )
    overview = summarize_partition_overview(summaries)

    assert overview["num_clients"] == 4
    assert overview["total_samples"] == len(targets)
    assert overview["min_client_samples"] == 20
    assert overview["max_client_samples"] == 20
