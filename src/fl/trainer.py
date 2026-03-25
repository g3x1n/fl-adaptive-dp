"""Shared local training and evaluation loops."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.optim import compute_proximal_term
from src.privacy import clip_and_add_noise


def train_one_epoch(
    model: nn.Module,
    dataloader: Any,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    algorithm: str = "fedavg",
    global_model: nn.Module | None = None,
    proximal_mu: float = 0.0,
    dp_mode: str = "none",
    clip_norm: float = 1.0,
    noise_multiplier: float = 0.0,
) -> dict[str, float]:
    """Run one local epoch and return the average training loss.

    FedProx reuses the standard supervised loss and adds a proximal penalty
    that discourages local weights from drifting too far from the global model.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    pre_clip_norm_sum = 0.0
    post_clip_norm_sum = 0.0
    num_batches = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        if algorithm.lower() == "fedprox":
            if global_model is None:
                raise ValueError("FedProx training requires the global model reference.")
            loss = loss + compute_proximal_term(
                local_model=model,
                global_model=global_model,
                proximal_mu=proximal_mu,
            )
        loss.backward()

        grad_stats = {
            "pre_clip_grad_norm": 0.0,
            "post_clip_grad_norm": 0.0,
        }
        if dp_mode in {"fixed", "adaptive"}:
            grad_stats = clip_and_add_noise(
                model=model,
                clip_norm=clip_norm,
                noise_multiplier=noise_multiplier,
            )

        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        pre_clip_norm_sum += grad_stats["pre_clip_grad_norm"]
        post_clip_norm_sum += grad_stats["post_clip_grad_norm"]
        num_batches += 1

    return {
        "loss": total_loss / max(total_samples, 1),
        "avg_pre_clip_grad_norm": pre_clip_norm_sum / max(num_batches, 1),
        "avg_post_clip_grad_norm": post_clip_norm_sum / max(num_batches, 1),
    }


def count_optimizer_steps(dataloader: Any, local_epochs: int) -> int:
    """Estimate how many optimizer steps a client performs in one round."""
    return len(dataloader) * local_epochs


def evaluate_model(model: nn.Module, dataloader: Any, device: torch.device) -> dict[str, float]:
    """Evaluate the global model on a held-out dataset."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)
            predictions = logits.argmax(dim=1)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == targets).sum().item()
            total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }
