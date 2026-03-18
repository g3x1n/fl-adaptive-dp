"""Experiment runner for the minimal FedAvg/FedProx training loop."""

from __future__ import annotations

from torch.utils.data import DataLoader, Subset

from src.data import (
    build_client_dataloaders,
    build_client_subsets,
    extract_targets,
    load_dataset,
    partition_dataset,
    summarize_client_distributions,
    summarize_partition_overview,
)
from src.fl.client import LocalClient
from src.fl.server import SynchronousFLServer
from src.models import build_model
from src.utils.device import resolve_device


def _maybe_limit_dataset(dataset, max_samples: int | None):
    """Optionally cap a dataset size for quick debug runs."""
    if max_samples is None:
        return dataset
    capped_size = min(len(dataset), max_samples)
    return Subset(dataset, list(range(capped_size)))


def run_federated_experiment(config: dict, logger, metrics_writer) -> dict:
    """Build datasets, clients and model, then execute the minimal FL loop."""
    dataset_config = config["dataset"]
    runtime_config = config["runtime"]

    train_dataset = load_dataset(
        dataset_name=dataset_config["name"],
        root=dataset_config["root"],
        train=True,
        download=dataset_config["download"],
    )
    test_dataset = load_dataset(
        dataset_name=dataset_config["name"],
        root=dataset_config["root"],
        train=False,
        download=dataset_config["download"],
    )

    train_dataset = _maybe_limit_dataset(train_dataset, dataset_config["max_train_samples"])
    test_dataset = _maybe_limit_dataset(test_dataset, dataset_config["max_test_samples"])

    # When a debug config uses a capped Subset, we need to remap labels so the
    # partitioner only sees the active samples rather than the full dataset.
    targets = extract_targets(train_dataset.dataset if isinstance(train_dataset, Subset) else train_dataset)
    if isinstance(train_dataset, Subset):
        targets = [targets[index] for index in train_dataset.indices]

    partition_map = partition_dataset(
        targets=targets,
        mode=dataset_config["partition_mode"],
        num_clients=dataset_config["num_clients"],
        seed=config["experiment"]["seed"],
        alpha=dataset_config["dirichlet_alpha"],
    )

    client_summaries = summarize_client_distributions(
        targets=targets,
        partition_map=partition_map,
        num_classes=dataset_config["num_classes"],
    )
    overview = summarize_partition_overview(client_summaries)
    logger.info("Partition overview: %s", overview)
    logger.info("First client summaries: %s", client_summaries[:3])

    client_subsets = build_client_subsets(train_dataset, partition_map)
    client_loaders = build_client_dataloaders(
        client_subsets=client_subsets,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=runtime_config["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["eval_batch_size"],
        shuffle=False,
        num_workers=runtime_config["num_workers"],
    )

    device = resolve_device(runtime_config["device"])
    global_model = build_model(
        model_name=config["model"]["name"],
        dataset_name=dataset_config["name"],
        num_classes=dataset_config["num_classes"],
    ).to(device)

    clients = [
        # Each simulated client owns exactly one local DataLoader.
        LocalClient(client_id=client_id, dataloader=dataloader, device=device)
        for client_id, dataloader in client_loaders.items()
    ]

    server = SynchronousFLServer(
        global_model=global_model,
        clients=clients,
        test_loader=test_loader,
        device=device,
        metrics_writer=metrics_writer,
        logger=logger,
        config=config,
    )
    return server.run()
