"""FedAvg aggregation utilities."""

from __future__ import annotations

from collections import OrderedDict

import torch


def aggregate_fedavg(
    client_states: list[OrderedDict[str, torch.Tensor]],
    client_weights: list[int],
) -> OrderedDict[str, torch.Tensor]:
    """Compute a weighted average over client model parameters."""
    if not client_states:
        raise ValueError("FedAvg aggregation requires at least one client state.")

    total_weight = float(sum(client_weights))
    aggregated_state: OrderedDict[str, torch.Tensor] = OrderedDict()

    for key in client_states[0].keys():
        weighted_sum = sum(
            state[key].float() * (weight / total_weight)
            for state, weight in zip(client_states, client_weights)
        )
        aggregated_state[key] = weighted_sum

    return aggregated_state


def aggregate_weighted_updates(
    client_updates: list[OrderedDict[str, torch.Tensor]],
    client_weights: list[int],
) -> OrderedDict[str, torch.Tensor]:
    """Compute the weighted average over client update tensors."""
    if not client_updates:
        raise ValueError("Weighted update aggregation requires at least one client update.")

    total_weight = float(sum(client_weights))
    aggregated_update: OrderedDict[str, torch.Tensor] = OrderedDict()

    for key in client_updates[0].keys():
        weighted_sum = sum(
            update[key].float() * (weight / total_weight)
            for update, weight in zip(client_updates, client_weights)
        )
        aggregated_update[key] = weighted_sum

    return aggregated_update
