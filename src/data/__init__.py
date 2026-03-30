"""Dataset loading, partitioning and client statistics helpers."""

from src.data.datasets import (
    build_client_dataloaders,
    build_client_subsets,
    extract_targets,
    get_num_classes,
    load_dataset,
)
from src.data.partition import partition_dataset, partition_dirichlet, partition_iid
from src.data.statistics import summarize_client_distributions, summarize_partition_overview

__all__ = [
    "build_client_dataloaders",
    "build_client_subsets",
    "extract_targets",
    "get_num_classes",
    "load_dataset",
    "partition_dataset",
    "partition_dirichlet",
    "partition_iid",
    "summarize_client_distributions",
    "summarize_partition_overview",
]
