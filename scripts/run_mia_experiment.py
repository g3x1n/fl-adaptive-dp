"""Train a federated model and evaluate black-box membership inference."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    build_client_dataloaders,
    build_client_subsets,
    extract_targets,
    partition_dataset,
    load_dataset,
    summarize_client_distributions,
    summarize_partition_overview,
)
from src.evaluation.logger import JsonlWriter, MetricsWriter, default_round_metric_fieldnames, setup_logger
from src.fl.client import LocalClient
from src.fl.server import SynchronousFLServer
from src.models import build_model
from src.privacy import run_membership_inference_attack
from src.privacy.mia import sample_dataset_subset
from src.utils.config import flatten_config, load_config
from src.utils.device import configure_runtime_backend, resolve_dataloader_kwargs, resolve_device
from src.utils.io import dump_json, dump_yaml, ensure_dir, timestamp
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run federated training plus membership inference attack.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML experiment config.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional custom run directory suffix.")
    parser.add_argument("--member-samples", type=int, default=2000, help="Number of member samples for attack.")
    parser.add_argument(
        "--nonmember-samples",
        type=int,
        default=2000,
        help="Number of non-member samples for attack.",
    )
    parser.add_argument("--attack-batch-size", type=int, default=512, help="Batch size for attack scoring.")
    return parser.parse_args()


def _maybe_limit_dataset(dataset, max_samples: int | None):
    if max_samples is None:
        return dataset
    capped_size = min(len(dataset), max_samples)
    return Subset(dataset, list(range(capped_size)))


def _build_run_dir(config: dict, run_name: str | None) -> Path:
    base_name = run_name or f"{config['experiment']['name']}_mia"
    run_dir = Path(config["experiment"]["output_root"]) / f"{timestamp()}_{base_name}"
    ensure_dir(run_dir)
    ensure_dir(run_dir / "figures")
    return run_dir


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_global_seed(config["experiment"]["seed"])

    run_dir = _build_run_dir(config, args.run_name)
    logger = setup_logger("mia_experiment", run_dir / "train.log")

    dump_yaml(run_dir / "config.yaml", config)
    dump_json(
        run_dir / "summary.json",
        {
            "status": "initialized",
            "run_dir": str(run_dir.resolve()),
            "algorithm": config["training"]["algorithm"],
            "dataset": config["dataset"]["name"],
        },
    )

    metrics_writer = MetricsWriter(
        run_dir / "metrics.csv",
        fieldnames=default_round_metric_fieldnames(),
    )
    round_jsonl_writer = JsonlWriter(run_dir / "round_summary.jsonl")
    client_jsonl_writer = JsonlWriter(run_dir / "client_metrics.jsonl")
    metrics_writer.write(
        {
            "round": 0,
            "learning_rate": None,
            "selected_clients": config["dataset"]["num_clients"],
            "algorithm": config["training"]["algorithm"],
            "dp_mode": config["privacy"]["dp_mode"],
            "compression_mode": config["compression"]["mode"],
            "train_loss": None,
            "test_loss": None,
            "test_accuracy": None,
            "best_test_accuracy": None,
            "epsilon_spent": None,
            "upload_payload_bytes": None,
            "pre_compression_payload_bytes": None,
            "noise_multiplier": None,
            "noise_multiplier_min": None,
            "noise_multiplier_max": None,
            "noise_multiplier_median": None,
            "clip_norm": None,
            "clip_norm_min": None,
            "clip_norm_max": None,
            "clip_norm_median": None,
            "schedule_reason": None,
            "compression_ratio": None,
            "nnz_params": None,
            "avg_update_norm": None,
            "median_update_norm": None,
            "max_update_norm": None,
            "avg_grad_norm": None,
            "median_grad_norm": None,
            "max_grad_norm": None,
            "median_client_loss": None,
            "max_client_loss": None,
            "median_client_epsilon": None,
            "max_client_epsilon": None,
        }
    )

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
    logger.info("Partition overview: %s", summarize_partition_overview(client_summaries))
    logger.info("First client summaries: %s", client_summaries[:3])
    logger.info("Flattened config: %s", flatten_config(config))

    client_subsets = build_client_subsets(train_dataset, partition_map)
    device = resolve_device(runtime_config["device"])
    configure_runtime_backend(device, runtime_config)
    dataloader_kwargs = resolve_dataloader_kwargs(runtime_config, device)
    client_loaders = build_client_dataloaders(
        client_subsets=client_subsets,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=dataloader_kwargs["num_workers"],
        pin_memory=dataloader_kwargs["pin_memory"],
        persistent_workers=dataloader_kwargs["persistent_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["eval_batch_size"],
        shuffle=False,
        num_workers=dataloader_kwargs["num_workers"],
        pin_memory=dataloader_kwargs["pin_memory"],
        persistent_workers=dataloader_kwargs["persistent_workers"],
    )
    global_model = build_model(
        model_name=config["model"]["name"],
        dataset_name=dataset_config["name"],
        num_classes=dataset_config["num_classes"],
    ).to(device)
    clients = [
        LocalClient(client_id=client_id, dataloader=dataloader, device=device)
        for client_id, dataloader in client_loaders.items()
    ]
    server = SynchronousFLServer(
        global_model=global_model,
        clients=clients,
        test_loader=test_loader,
        device=device,
        metrics_writer=metrics_writer,
        round_jsonl_writer=round_jsonl_writer,
        client_jsonl_writer=client_jsonl_writer,
        logger=logger,
        config=config,
    )
    train_results = server.run()

    member_dataset = sample_dataset_subset(
        train_dataset,
        size=args.member_samples,
        seed=config["experiment"]["seed"],
    )
    nonmember_dataset = sample_dataset_subset(
        test_dataset,
        size=args.nonmember_samples,
        seed=config["experiment"]["seed"] + 1,
    )

    attack_result = run_membership_inference_attack(
        model=global_model,
        member_dataset=member_dataset,
        nonmember_dataset=nonmember_dataset,
        batch_size=args.attack_batch_size,
        device=device,
        output_csv_path=run_dir / "mia_scores.csv",
    )

    summary = {
        "status": "completed",
        "run_dir": str(run_dir.resolve()),
        "algorithm": config["training"]["algorithm"],
        "dataset": config["dataset"]["name"],
        "results": train_results,
        "membership_inference": {
            "member_samples": len(member_dataset),
            "nonmember_samples": len(nonmember_dataset),
            "auc_loss": round(attack_result.auc_loss, 6),
            "auc_confidence": round(attack_result.auc_confidence, 6),
            "best_acc_loss": round(attack_result.best_acc_loss, 6),
            "best_acc_confidence": round(attack_result.best_acc_confidence, 6),
            "member_mean_loss_score": round(attack_result.member_mean_loss_score, 6),
            "nonmember_mean_loss_score": round(attack_result.nonmember_mean_loss_score, 6),
            "member_mean_confidence": round(attack_result.member_mean_confidence, 6),
            "nonmember_mean_confidence": round(attack_result.nonmember_mean_confidence, 6),
        },
    }
    dump_json(run_dir / "summary.json", summary)
    logger.info("MIA summary: %s", summary["membership_inference"])


if __name__ == "__main__":
    main()
