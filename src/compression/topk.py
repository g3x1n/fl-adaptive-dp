"""Top-k update compression utilities."""

from __future__ import annotations

from collections import OrderedDict

import torch


def compute_model_update(
    global_state: OrderedDict[str, torch.Tensor],
    local_state: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Build a client update as the local-global parameter delta."""
    return OrderedDict(
        (name, local_state[name].float() - global_state[name].float())
        for name in global_state.keys()
    )


def apply_model_update(
    global_state: OrderedDict[str, torch.Tensor],
    update: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Apply an aggregated client update to the global model state."""
    return OrderedDict(
        (name, global_state[name].float() + update[name].float())
        for name in global_state.keys()
    )


def compress_topk(
    update: OrderedDict[str, torch.Tensor],
    topk_ratio: float,
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, float]]:
    """Keep only the largest-magnitude update entries per tensor."""
    total_params = 0
    nnz_params = 0
    compressed_update: OrderedDict[str, torch.Tensor] = OrderedDict()

    for name, tensor in update.items():
        flat = tensor.flatten()
        total_params += flat.numel()

        if topk_ratio >= 1.0 or flat.numel() == 0:
            kept = flat.numel()
            compressed_tensor = tensor.clone()
        else:
            k = max(1, int(round(flat.numel() * topk_ratio)))
            _, topk_indices = torch.topk(flat.abs(), k=k, largest=True, sorted=False)
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask[topk_indices] = True
            sparse_flat = torch.where(mask, flat, torch.zeros_like(flat))
            compressed_tensor = sparse_flat.view_as(tensor)
            kept = k

        nnz_params += kept
        compressed_update[name] = compressed_tensor

    pre_bytes = sum(tensor.numel() * tensor.element_size() for tensor in update.values())
    # Payload estimate includes indices for each retained element.
    estimated_payload_bytes = nnz_params * (4 + 8)
    compression_ratio = nnz_params / max(total_params, 1)

    return compressed_update, {
        "total_params": float(total_params),
        "nnz_params": float(nnz_params),
        "compression_ratio": float(compression_ratio),
        "pre_compression_payload_bytes": float(pre_bytes),
        "upload_payload_bytes": float(estimated_payload_bytes),
    }
