"""Run server-side white-box gradient inversion attacks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_client_subsets, extract_targets, load_dataset, partition_dataset
from src.models import build_model
from src.privacy import (
    collect_observed_gradients,
    evaluate_reconstruction,
    reconstruct_single_sample_from_gradients,
)
from src.utils.config import load_config
from src.utils.device import configure_runtime_backend, resolve_device
from src.utils.io import dump_json, dump_yaml, ensure_dir, timestamp
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a white-box gradient inversion attack.")
    parser.add_argument("--baseline-config", type=str, required=True, help="No-DP config path.")
    parser.add_argument("--dp-config", type=str, required=True, help="DP config path.")
    parser.add_argument("--client-id", type=int, default=0, help="Target client id.")
    parser.add_argument("--sample-offset", type=int, default=0, help="Index inside the target client's subset.")
    parser.add_argument("--steps", type=int, default=300, help="Optimization steps for inversion.")
    parser.add_argument("--lr", type=float, default=0.1, help="Dummy input learning rate.")
    parser.add_argument("--tv-weight", type=float, default=1e-4, help="Total variation regularization weight.")
    parser.add_argument("--noise-seed", type=int, default=42, help="Noise seed for DP perturbation.")
    parser.add_argument("--device", type=str, default="auto", help="Attack device: auto/cuda/cuda:0/cpu.")
    return parser.parse_args()


def _prepare_target_sample(config: dict, client_id: int, sample_offset: int):
    dataset_config = config["dataset"]
    train_dataset = load_dataset(
        dataset_name=dataset_config["name"],
        root=dataset_config["root"],
        train=True,
        download=dataset_config["download"],
    )
    targets = extract_targets(train_dataset)
    partition_map = partition_dataset(
        targets=targets,
        mode=dataset_config["partition_mode"],
        num_clients=dataset_config["num_clients"],
        seed=config["experiment"]["seed"],
        alpha=dataset_config["dirichlet_alpha"],
    )
    client_subsets = build_client_subsets(train_dataset, partition_map)
    client_subset = client_subsets[client_id]
    if sample_offset >= len(client_subset):
        raise IndexError(f"sample_offset={sample_offset} exceeds client subset size {len(client_subset)}")
    sample_input, sample_label = client_subset[sample_offset]
    return sample_input.unsqueeze(0), int(sample_label)


def _save_mnist_tensor(image: torch.Tensor, path: Path) -> None:
    # MNIST tensors are normalized with mean=0.1307, std=0.3081.
    denorm = image.clone()
    denorm = denorm * 0.3081 + 0.1307
    denorm = denorm.clamp(0.0, 1.0)
    save_image(denorm, str(path))


def _attack_one_setting(
    *,
    config: dict,
    sample_input: torch.Tensor,
    sample_label: int,
    dp_mode: str,
    clip_norm: float,
    noise_multiplier: float,
    steps: int,
    learning_rate: float,
    tv_weight: float,
    noise_seed: int,
    device: torch.device,
) -> tuple[dict, torch.Tensor]:
    model = build_model(
        model_name=config["model"]["name"],
        dataset_name=config["dataset"]["name"],
        num_classes=config["dataset"]["num_classes"],
    ).to(device)
    inputs = sample_input.to(device)
    targets = torch.tensor([sample_label], device=device)

    observed_gradients = collect_observed_gradients(
        model=model,
        inputs=inputs,
        targets=targets,
        dp_mode=dp_mode,
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
        noise_seed=noise_seed,
    )
    reconstructed_input, recovered_label, final_gradient_loss = reconstruct_single_sample_from_gradients(
        model=model,
        observed_gradients=observed_gradients,
        input_shape=tuple(inputs.shape[1:]),
        device=device,
        steps=steps,
        learning_rate=learning_rate,
        tv_weight=tv_weight,
    )
    attack_result = evaluate_reconstruction(
        original_input=inputs.squeeze(0),
        original_label=sample_label,
        reconstructed_input=reconstructed_input.squeeze(0),
        recovered_label=recovered_label,
        final_gradient_loss=final_gradient_loss,
    )
    return {
        "dp_mode": dp_mode,
        "clip_norm": clip_norm,
        "noise_multiplier": noise_multiplier,
        "recovered_label": attack_result.recovered_label,
        "label_match": attack_result.label_match,
        "reconstruction_mse": round(attack_result.reconstruction_mse, 6),
        "reconstruction_psnr": round(attack_result.reconstruction_psnr, 6),
        "final_gradient_loss": round(attack_result.final_gradient_loss, 6),
    }, reconstructed_input


def main() -> None:
    args = parse_args()
    baseline_config = load_config(args.baseline_config)
    dp_config = load_config(args.dp_config)
    set_global_seed(baseline_config["experiment"]["seed"])
    attack_device = resolve_device(args.device)
    configure_runtime_backend(attack_device, baseline_config.get("runtime", {}))

    run_dir = Path(baseline_config["experiment"]["output_root"]) / f"{timestamp()}_whitebox_attack"
    ensure_dir(run_dir)
    dump_yaml(run_dir / "baseline_config.yaml", baseline_config)
    dump_yaml(run_dir / "dp_config.yaml", dp_config)

    sample_input, sample_label = _prepare_target_sample(
        baseline_config,
        client_id=args.client_id,
        sample_offset=args.sample_offset,
    )

    baseline_summary, baseline_reconstruction = _attack_one_setting(
        config=baseline_config,
        sample_input=sample_input,
        sample_label=sample_label,
        dp_mode="none",
        clip_norm=1.0,
        noise_multiplier=0.0,
        steps=args.steps,
        learning_rate=args.lr,
        tv_weight=args.tv_weight,
        noise_seed=args.noise_seed,
        device=attack_device,
    )

    dp_privacy = dp_config["privacy"]
    dp_noise_multiplier = float(
        dp_privacy.get("min_noise_multiplier", dp_privacy.get("noise_multiplier", 0.0))
    )
    dp_summary, dp_reconstruction = _attack_one_setting(
        config=dp_config,
        sample_input=sample_input,
        sample_label=sample_label,
        dp_mode=str(dp_privacy["dp_mode"]),
        clip_norm=float(dp_privacy.get("clip_norm", 1.0)),
        noise_multiplier=dp_noise_multiplier,
        steps=args.steps,
        learning_rate=args.lr,
        tv_weight=args.tv_weight,
        noise_seed=args.noise_seed,
        device=attack_device,
    )

    _save_mnist_tensor(sample_input, run_dir / "target.png")
    _save_mnist_tensor(baseline_reconstruction, run_dir / "reconstruction_nodp.png")
    _save_mnist_tensor(dp_reconstruction, run_dir / "reconstruction_adaptive_dp.png")

    summary = {
        "target_client_id": args.client_id,
        "target_sample_offset": args.sample_offset,
        "target_label": sample_label,
        "attack_steps": args.steps,
        "attack_learning_rate": args.lr,
        "attack_tv_weight": args.tv_weight,
        "baseline_attack": baseline_summary,
        "adaptive_dp_attack": dp_summary,
    }
    dump_json(run_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
