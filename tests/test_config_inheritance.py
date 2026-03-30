"""Tests for inherited final experiment configs."""

from __future__ import annotations

from src.utils.config import load_config


def test_final_config_inherits_runtime_dataset_and_method_settings() -> None:
    config = load_config(
        "configs/experiments/final/exp_a_noniid_alpha_0_1/mnist/fednova_adaptive_dp.yaml"
    )

    assert config["dataset"]["name"] == "mnist"
    assert config["dataset"]["dirichlet_alpha"] == 0.1
    assert config["training"]["algorithm"] == "fednova"
    assert config["privacy"]["dp_mode"] == "adaptive"
    assert config["runtime"]["num_workers"] == 8
