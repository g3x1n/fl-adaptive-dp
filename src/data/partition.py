"""IID and Dirichlet-based partition helpers."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def partition_iid(targets: list[int], num_clients: int, seed: int) -> dict[int, list[int]]:
    """Shuffle all samples uniformly and split them evenly across clients."""
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(targets))
    rng.shuffle(indices)
    chunks = np.array_split(indices, num_clients)
    return {
        client_id: sorted(chunk.astype(int).tolist())
        for client_id, chunk in enumerate(chunks)
    }


def partition_dirichlet(
    targets: list[int],
    num_clients: int,
    alpha: float,
    seed: int,
) -> dict[int, list[int]]:
    """Partition labels by class using a Dirichlet allocation."""
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if alpha <= 0:
        raise ValueError("alpha must be positive for Dirichlet partitioning.")

    rng = np.random.default_rng(seed)
    targets_array = np.asarray(targets)
    unique_classes = np.unique(targets_array)
    client_indices: dict[int, list[int]] = defaultdict(list)

    for class_id in unique_classes:
        class_indices = np.flatnonzero(targets_array == class_id)
        rng.shuffle(class_indices)

        class_proportions = rng.dirichlet(np.full(num_clients, alpha))
        split_points = (np.cumsum(class_proportions)[:-1] * len(class_indices)).astype(int)
        class_splits = np.split(class_indices, split_points)

        for client_id, split in enumerate(class_splits):
            client_indices[client_id].extend(split.astype(int).tolist())

    return {
        client_id: sorted(client_indices.get(client_id, []))
        for client_id in range(num_clients)
    }


def partition_dataset(
    targets: list[int],
    mode: str,
    num_clients: int,
    seed: int,
    alpha: float | None = None,
) -> dict[int, list[int]]:
    """Dispatch to the configured partition strategy."""
    normalized_mode = mode.lower()
    if normalized_mode == "iid":
        return partition_iid(targets=targets, num_clients=num_clients, seed=seed)
    if normalized_mode == "dirichlet":
        return partition_dirichlet(
            targets=targets,
            num_clients=num_clients,
            alpha=0.5 if alpha is None else float(alpha),
            seed=seed,
        )
    raise ValueError(f"Unsupported partition mode: {mode}")
