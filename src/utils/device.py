"""Device and runtime backend helpers."""

from __future__ import annotations

from typing import Any

import torch


def resolve_device(device_name: str) -> torch.device:
    """Resolve a config value into a concrete torch device."""
    normalized = str(device_name).lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(normalized)


def configure_runtime_backend(device: torch.device, runtime_config: dict[str, Any]) -> None:
    """Apply backend settings that are especially useful on CUDA hosts."""
    if device.type != "cuda":
        return

    allow_tf32 = bool(runtime_config.get("allow_tf32", True))
    cudnn_benchmark = bool(runtime_config.get("cudnn_benchmark", True))
    matmul_precision = str(runtime_config.get("matmul_precision", "high")).lower()

    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = cudnn_benchmark

    if matmul_precision in {"highest", "high", "medium"}:
        torch.set_float32_matmul_precision(matmul_precision)


def resolve_dataloader_kwargs(runtime_config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Resolve loader flags that differ between CPU/MPS/CUDA hosts."""
    num_workers = int(runtime_config.get("num_workers", 0))

    pin_memory_setting = runtime_config.get("pin_memory", "auto")
    if isinstance(pin_memory_setting, str) and pin_memory_setting.lower() == "auto":
        pin_memory = device.type == "cuda"
    else:
        pin_memory = bool(pin_memory_setting)

    persistent_setting = runtime_config.get("persistent_workers", "auto")
    if isinstance(persistent_setting, str) and persistent_setting.lower() == "auto":
        persistent_workers = num_workers > 0
    else:
        persistent_workers = bool(persistent_setting)
    persistent_workers = persistent_workers and num_workers > 0

    return {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
