"""Tests for Top-k update compression."""

from __future__ import annotations

from collections import OrderedDict

import torch

from src.compression import apply_model_update, compress_topk, compute_model_update


def test_topk_ratio_one_matches_identity() -> None:
    update = OrderedDict({"weight": torch.tensor([1.0, -2.0, 3.0])})
    compressed, stats = compress_topk(update, topk_ratio=1.0)

    assert torch.allclose(compressed["weight"], update["weight"])
    assert stats["compression_ratio"] == 1.0


def test_topk_zeroes_out_small_entries() -> None:
    update = OrderedDict({"weight": torch.tensor([1.0, -5.0, 2.0, 4.0])})
    compressed, stats = compress_topk(update, topk_ratio=0.5)

    assert stats["nnz_params"] == 2.0
    assert torch.count_nonzero(compressed["weight"]).item() == 2


def test_apply_model_update_restores_local_state_from_delta() -> None:
    global_state = OrderedDict({"weight": torch.tensor([1.0, 2.0])})
    local_state = OrderedDict({"weight": torch.tensor([1.5, 1.0])})

    update = compute_model_update(global_state=global_state, local_state=local_state)
    restored = apply_model_update(global_state=global_state, update=update)

    assert torch.allclose(restored["weight"], local_state["weight"])
