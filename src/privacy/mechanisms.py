"""Lightweight DP mechanisms used by the local client training loop."""

from __future__ import annotations

import math

import torch
from torch import nn


def compute_global_grad_norm(model: nn.Module) -> float:
    """Measure the L2 norm over all parameter gradients."""
    squared_norm = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad_norm = parameter.grad.detach().data.norm(2).item()
        squared_norm += grad_norm**2
    return math.sqrt(squared_norm)


def clip_and_add_noise(
    model: nn.Module,
    clip_norm: float,
    noise_multiplier: float,
) -> dict[str, float]:
    """Clip gradients and optionally inject Gaussian noise.

    The implementation is intentionally explicit so it is easy to audit in
    experiments and thesis writeups.
    """
    pre_clip_grad_norm = compute_global_grad_norm(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
    post_clip_grad_norm = compute_global_grad_norm(model)

    if noise_multiplier > 0:
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            noise = torch.randn_like(parameter.grad) * (noise_multiplier * clip_norm)
            parameter.grad.add_(noise)

    return {
        "pre_clip_grad_norm": pre_clip_grad_norm,
        "post_clip_grad_norm": post_clip_grad_norm,
    }
