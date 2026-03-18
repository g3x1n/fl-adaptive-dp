"""Device selection helpers."""

from __future__ import annotations

import torch


def resolve_device(device_name: str) -> torch.device:
    """Resolve a config value into a concrete torch device."""
    normalized = device_name.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(normalized)

