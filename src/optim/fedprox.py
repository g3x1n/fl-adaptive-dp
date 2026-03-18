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

    penalty = torch.zeros(1, device=next(local_model.parameters()).device).squeeze()
    for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
        penalty = penalty + torch.sum((local_param - global_param.detach()) ** 2)
    return 0.5 * proximal_mu * penalty

