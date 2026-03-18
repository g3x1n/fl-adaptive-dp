"""Inspect dataset loading and federated partition statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    extract_targets,
    get_num_classes,
    load_dataset,
    partition_dataset,
    summarize_client_distributions,
    summarize_partition_overview,
)
from src.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect federated dataset partitions.")
    parser.add_argument("--config", type=str, required=True, help="Experiment config path.")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Inspect the training split. The default is the training split already.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_config = config["dataset"]

    dataset = load_dataset(
        dataset_name=dataset_config["name"],
        root=dataset_config["root"],
        train=True,
        download=dataset_config["download"],
    )
    targets = extract_targets(dataset)
    partition_map = partition_dataset(
        targets=targets,
        mode=dataset_config["partition_mode"],
        num_clients=dataset_config["num_clients"],
        seed=config["experiment"]["seed"],
        alpha=dataset_config["dirichlet_alpha"],
    )

    summaries = summarize_client_distributions(
        targets=targets,
        partition_map=partition_map,
        num_classes=get_num_classes(dataset_config["name"]),
    )
    overview = summarize_partition_overview(summaries)

    print(
        json.dumps(
            {
                "dataset": dataset_config["name"],
                "partition_mode": dataset_config["partition_mode"],
                "overview": overview,
                "first_clients": summaries[:3],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

