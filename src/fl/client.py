"""Local client implementation for simulated federated training."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import torch
from torch import nn

from src.fl.trainer import count_optimizer_steps, train_one_epoch
from src.compression import compress_topk, compute_model_update
from src.privacy import PrivacyAccountant


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
        self.privacy_accountant = PrivacyAccountant()

    def fit(
        self,
        global_model: nn.Module,
        local_epochs: int,
        learning_rate: float,
        algorithm: str,
        proximal_mu: float = 0.0,
        optimizer_config: dict | None = None,
        privacy_plan: dict | None = None,
        compression_config: dict | None = None,
    ) -> dict:
        """Train a copy of the global model and return an auditable client payload."""
        local_model = deepcopy(global_model).to(self.device)
        reference_global_model = deepcopy(global_model).to(self.device)
        reference_global_model.eval()
        optimizer_config = optimizer_config or {}
        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=learning_rate,
            momentum=float(optimizer_config.get("momentum", 0.0)),
            weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
            nesterov=bool(optimizer_config.get("nesterov", False)),
        )
        criterion = nn.CrossEntropyLoss()
        privacy_plan = privacy_plan or {}
        compression_config = compression_config or {}

        epoch_stats = []
        for _ in range(local_epochs):
            epoch_stat = train_one_epoch(
                model=local_model,
                dataloader=self.dataloader,
                optimizer=optimizer,
                criterion=criterion,
                device=self.device,
                algorithm=algorithm,
                global_model=reference_global_model,
                proximal_mu=proximal_mu,
                dp_mode=privacy_plan.get("dp_mode", "none"),
                clip_norm=privacy_plan.get("clip_norm", 1.0),
                noise_multiplier=privacy_plan.get("noise_multiplier", 0.0),
            )
            epoch_stats.append(epoch_stat)

        local_state = OrderedDict(
            (name, tensor.detach().cpu().clone())
            for name, tensor in local_model.state_dict().items()
        )
        global_state = OrderedDict(
            (name, tensor.detach().cpu().clone())
            for name, tensor in reference_global_model.state_dict().items()
        )
        client_update = compute_model_update(global_state=global_state, local_state=local_state)

        compression_mode = str(compression_config.get("mode", "none")).lower()
        if compression_mode == "topk" and compression_config.get("compress_updates", True):
            compressed_update, compression_stats = compress_topk(
                update=client_update,
                topk_ratio=float(compression_config.get("topk_ratio", 1.0)),
            )
        else:
            compressed_update = client_update
            total_params = sum(tensor.numel() for tensor in client_update.values())
            pre_bytes = sum(tensor.numel() * tensor.element_size() for tensor in client_update.values())
            compression_stats = {
                "total_params": float(total_params),
                "nnz_params": float(total_params),
                "compression_ratio": 1.0,
                "pre_compression_payload_bytes": float(pre_bytes),
                "upload_payload_bytes": float(pre_bytes),
            }

        num_samples = len(self.dataloader.dataset)
        local_steps = count_optimizer_steps(self.dataloader, local_epochs)
        avg_loss = sum(stat["loss"] for stat in epoch_stats) / max(len(epoch_stats), 1)
        avg_pre_clip_norm = sum(stat["avg_pre_clip_grad_norm"] for stat in epoch_stats) / max(len(epoch_stats), 1)
        avg_post_clip_norm = sum(stat["avg_post_clip_grad_norm"] for stat in epoch_stats) / max(len(epoch_stats), 1)
        sample_rate = min(1.0, compression_config.get("batch_size", 1) / max(num_samples, 1))

        epsilon_spent = self.privacy_accountant.epsilon_spent
        if privacy_plan.get("dp_mode") in {"fixed", "adaptive"}:
            epsilon_spent = self.privacy_accountant.step(
                sample_rate=sample_rate,
                noise_multiplier=float(privacy_plan.get("noise_multiplier", 0.0)),
                delta=float(privacy_plan.get("delta", 1e-5)),
                num_steps=local_steps,
            )

        update_l2_norm = torch.sqrt(
            sum(torch.sum(tensor.float() ** 2) for tensor in client_update.values())
        ).item()

        return {
            "update": compressed_update,
            "num_samples": num_samples,
            "avg_loss": avg_loss,
            "local_steps": local_steps,
            "epsilon_spent": epsilon_spent,
            "clip_norm": float(privacy_plan.get("clip_norm", 1.0)),
            "noise_multiplier": float(privacy_plan.get("noise_multiplier", 0.0)),
            "schedule_reason": str(privacy_plan.get("schedule_reason", "dp disabled")),
            "avg_pre_clip_grad_norm": avg_pre_clip_norm,
            "avg_post_clip_grad_norm": avg_post_clip_norm,
            "update_norm": update_l2_norm,
            **compression_stats,
        }
