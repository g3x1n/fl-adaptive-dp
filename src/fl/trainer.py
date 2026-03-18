"""Shared local training and evaluation loops."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.optim import compute_proximal_term


def train_one_epoch(
    model: nn.Module,
    dataloader: Any,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    algorithm: str = "fedavg",
    global_model: nn.Module | None = None,
    proximal_mu: float = 0.0,
) -> float:
    """Run one local epoch and return the average training loss.

    FedProx reuses the standard supervised loss and adds a proximal penalty
    that discourages local weights from drifting too far from the global model.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

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
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


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
