"""Tests for runtime device helpers."""

from __future__ import annotations

import torch

from src.utils.device import resolve_dataloader_kwargs


def test_cuda_like_runtime_enables_pin_memory_and_persistent_workers() -> None:
    runtime_config = {
        "num_workers": 4,
        "pin_memory": "auto",
        "persistent_workers": "auto",
    }
    kwargs = resolve_dataloader_kwargs(runtime_config, torch.device("cuda"))
    assert kwargs["pin_memory"] is True
    assert kwargs["persistent_workers"] is True


def test_cpu_runtime_disables_auto_pin_memory() -> None:
    runtime_config = {
        "num_workers": 0,
        "pin_memory": "auto",
        "persistent_workers": "auto",
    }
    kwargs = resolve_dataloader_kwargs(runtime_config, torch.device("cpu"))
    assert kwargs["pin_memory"] is False
    assert kwargs["persistent_workers"] is False
