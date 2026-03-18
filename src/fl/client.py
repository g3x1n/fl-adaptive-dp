"""Local client implementation for simulated federated training."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import torch
from torch import nn

from src.fl.trainer import count_optimizer_steps, train_one_epoch


class LocalClient:
    """A thin client wrapper that trains on its own local DataLoader."""

    def __init__(
        self,
        client_id: int,
        dataloader,
        device: torch.device,
    ) -> None:
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device

    def fit(
        self,
        global_model: nn.Module,
        local_epochs: int,
        learning_rate: float,
        algorithm: str,
        proximal_mu: float = 0.0,
    ) -> tuple[OrderedDict[str, torch.Tensor], int, float, int]:
        """Train a copy of the global model and return its updated parameters."""
        local_model = deepcopy(global_model).to(self.device)
        reference_global_model = deepcopy(global_model).to(self.device)
        reference_global_model.eval()
        optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        epoch_losses = []
        for _ in range(local_epochs):
            epoch_loss = train_one_epoch(
                model=local_model,
                dataloader=self.dataloader,
                optimizer=optimizer,
                criterion=criterion,
                device=self.device,
                algorithm=algorithm,
                global_model=reference_global_model,
                proximal_mu=proximal_mu,
            )
            epoch_losses.append(epoch_loss)

        state = OrderedDict(
            (name, tensor.detach().cpu().clone())
            for name, tensor in local_model.state_dict().items()
        )
        num_samples = len(self.dataloader.dataset)
        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        local_steps = count_optimizer_steps(self.dataloader, local_epochs)
        return state, num_samples, avg_loss, local_steps
