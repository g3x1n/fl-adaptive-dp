"""Filesystem and serialization helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def dump_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return output_path


def dump_yaml(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
    return output_path

