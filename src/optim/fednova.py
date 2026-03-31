"""FedNova aggregation utilities."""

from __future__ import annotations

from collections import OrderedDict

import torch


def aggregate_fednova(
    global_state: OrderedDict[str, torch.Tensor],
    client_states: list[OrderedDict[str, torch.Tensor]],
    client_weights: list[float],
    client_steps: list[int],
) -> OrderedDict[str, torch.Tensor]:
    """Aggregate client models using FedNova-style normalization.

    For vanilla SGD without momentum, the local normalization factor can be
    approximated by the number of optimizer steps performed by each client.
    """
    if not client_states:
        raise ValueError("FedNova aggregation requires at least one client state.")

    total_weight = float(sum(client_weights))
    normalized_global: OrderedDict[str, torch.Tensor] = OrderedDict()

    tau_eff = sum((weight / total_weight) * steps for weight, steps in zip(client_weights, client_steps))

    for key, global_tensor in global_state.items():
        aggregate_direction = torch.zeros_like(global_tensor, dtype=torch.float32)
        target_device = global_tensor.device

        for state, weight, steps in zip(client_states, client_weights, client_steps):
            # FedNova normalizes each local update by its effective local step count.
            local_tensor = state[key].to(device=target_device, dtype=torch.float32)
            local_direction = (global_tensor.float() - local_tensor) / max(float(steps), 1.0)
            aggregate_direction = aggregate_direction + (weight / total_weight) * local_direction

        normalized_global[key] = global_tensor.float() - tau_eff * aggregate_direction

    return normalized_global


def aggregate_fednova_updates(
    global_state: OrderedDict[str, torch.Tensor],
    client_updates: list[OrderedDict[str, torch.Tensor]],
    client_weights: list[float],
    client_steps: list[int],
) -> OrderedDict[str, torch.Tensor]:
    """Aggregate compressed client updates using FedNova normalization."""
    if not client_updates:
        raise ValueError("FedNova update aggregation requires at least one client update.")

    total_weight = float(sum(client_weights))
    aggregated_update: OrderedDict[str, torch.Tensor] = OrderedDict()
    tau_eff = sum((weight / total_weight) * steps for weight, steps in zip(client_weights, client_steps))

    for key in global_state.keys():
        target_tensor = global_state[key]
        target_device = target_tensor.device
        normalized_direction = torch.zeros_like(target_tensor, dtype=torch.float32)
        for update, weight, steps in zip(client_updates, client_weights, client_steps):
            local_direction = update[key].to(device=target_device, dtype=torch.float32) / max(float(steps), 1.0)
            normalized_direction = normalized_direction + (weight / total_weight) * local_direction
        aggregated_update[key] = tau_eff * normalized_direction

    return aggregated_update
