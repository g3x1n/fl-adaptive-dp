"""Client label-distribution summaries for audits and debugging."""

from __future__ import annotations


def summarize_client_distributions(
    targets: list[int],
    partition_map: dict[int, list[int]],
    num_classes: int,
) -> list[dict]:
    """Summarize label counts for each client partition."""
    summaries: list[dict] = []
    for client_id, sample_indices in sorted(partition_map.items()):
        class_counts = [0] * num_classes
        for sample_index in sample_indices:
            class_counts[int(targets[sample_index])] += 1

        total_samples = len(sample_indices)
        label_ratios = [
            (count / total_samples) if total_samples > 0 else 0.0
            for count in class_counts
        ]
        summaries.append(
            {
                "client_id": client_id,
                "num_samples": total_samples,
                "class_counts": class_counts,
                "class_ratios": label_ratios,
            }
        )
    return summaries


def summarize_partition_overview(client_summaries: list[dict]) -> dict:
    """Aggregate top-level statistics across client partitions."""
    sample_counts = [int(summary["num_samples"]) for summary in client_summaries]
    total_samples = sum(sample_counts)
    return {
        "num_clients": len(client_summaries),
        "total_samples": total_samples,
        "min_client_samples": min(sample_counts) if sample_counts else 0,
        "max_client_samples": max(sample_counts) if sample_counts else 0,
        "avg_client_samples": (total_samples / len(sample_counts)) if sample_counts else 0.0,
    }
