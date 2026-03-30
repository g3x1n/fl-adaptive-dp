"""Dataset and client dataloader helpers."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def _mnist_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def _cifar10_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )


def load_dataset(dataset_name: str, root: str, train: bool, download: bool) -> Dataset:
    """Load one of the supported torchvision datasets."""
    name = dataset_name.lower()
    if name == "mnist":
        return datasets.MNIST(root=root, train=train, download=download, transform=_mnist_transform())
    if name == "cifar10":
        return datasets.CIFAR10(root=root, train=train, download=download, transform=_cifar10_transform())
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_num_classes(dataset_name: str) -> int:
    name = dataset_name.lower()
    if name in {"mnist", "cifar10"}:
        return 10
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def extract_targets(dataset: Dataset) -> list[int]:
    """Extract integer labels from a torchvision dataset or a subset."""
    if isinstance(dataset, Subset):
        base_targets = extract_targets(dataset.dataset)
        return [base_targets[index] for index in dataset.indices]

    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError("Dataset does not expose `targets` for partitioning.")

    if isinstance(targets, torch.Tensor):
        return targets.cpu().tolist()
    return [int(target) for target in targets]


def build_client_subsets(dataset: Dataset, partition_map: Mapping[int, list[int]]) -> dict[int, Subset]:
    """Turn a client-to-indices mapping into dataset subsets."""
    return {
        client_id: Subset(dataset, indices)
        for client_id, indices in partition_map.items()
    }


def build_client_dataloaders(
    client_subsets: Mapping[int, Dataset],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> dict[int, DataLoader]:
    """Create one DataLoader per client subset."""
    return {
        client_id: DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        for client_id, subset in client_subsets.items()
    }
