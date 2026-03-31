"""FedProx utilities for local proximal regularization."""

from __future__ import annotations

import torch
from torch import nn


def compute_proximal_term(
    local_model: nn.Module,
    global_model: nn.Module,
    proximal_mu: float,
) -> torch.Tensor:
    """Compute the FedProx penalty between local and global parameters."""
    if proximal_mu <= 0:
        first_param = next(local_model.parameters())
        return torch.zeros(1, device=first_param.device, dtype=first_param.dtype).squeeze()

    first_param = next(local_model.parameters())
    penalty = torch.zeros(1, device=first_param.device, dtype=first_param.dtype).squeeze()
    for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
        reference_param = global_param.detach().to(device=local_param.device, dtype=local_param.dtype)
        penalty = penalty + torch.sum((local_param - reference_param) ** 2)
    return 0.5 * proximal_mu * penalty
