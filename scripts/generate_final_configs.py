"""Generate the final organized experiment config layout."""

from __future__ import annotations

from pathlib import Path
import shutil

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = ROOT / "configs" / "experiments"
FINAL_ROOT = EXPERIMENTS_ROOT / "final"
LEGACY_ROOT = EXPERIMENTS_ROOT / "legacy"


def dump_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def dump_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_common_configs() -> None:
    runtime = {
        "runtime": {
            "device": "auto",
            "num_workers": 8,
            "pin_memory": "auto",
            "persistent_workers": "auto",
            "allow_tf32": True,
            "cudnn_benchmark": True,
            "matmul_precision": "high",
        }
    }
    dump_yaml(FINAL_ROOT / "common" / "runtime_ubuntu.yaml", runtime)

    training_mnist = {
        "training": {
            "rounds": 100,
            "local_epochs": 2,
            "batch_size": 64,
            "eval_batch_size": 512,
            "learning_rate": 0.05,
            "lr_schedule": "exp",
            "lr_decay_gamma": 0.97,
            "lr_decay_step": 1,
            "min_learning_rate": 0.005,
            "fraction_fit": 1.0,
            "proximal_mu": 0.0,
        }
    }
    dump_yaml(FINAL_ROOT / "common" / "training" / "mnist_strong.yaml", training_mnist)

    training_cifar10 = {
        "training": {
            "rounds": 120,
            "local_epochs": 1,
            "batch_size": 64,
            "eval_batch_size": 512,
            "learning_rate": 0.02,
            "lr_schedule": "exp",
            "lr_decay_gamma": 0.98,
            "lr_decay_step": 1,
            "min_learning_rate": 0.001,
            "fraction_fit": 1.0,
            "proximal_mu": 0.0,
        }
    }
    dump_yaml(FINAL_ROOT / "common" / "training" / "cifar10_strong.yaml", training_cifar10)

    dataset_defs = {
        "mnist_iid": {
            "dataset": {
                "name": "mnist",
                "root": "data",
                "partition_mode": "iid",
                "num_clients": 10,
                "dirichlet_alpha": 0.1,
                "download": True,
                "num_classes": 10,
                "max_train_samples": None,
                "max_test_samples": None,
            },
            "model": {"name": "mnist_cnn"},
        },
        "mnist_noniid_alpha_1_0": {
            "dataset": {
                "name": "mnist",
                "root": "data",
                "partition_mode": "dirichlet",
                "num_clients": 10,
                "dirichlet_alpha": 1.0,
                "download": True,
                "num_classes": 10,
                "max_train_samples": None,
                "max_test_samples": None,
            },
            "model": {"name": "mnist_cnn"},
        },
        "mnist_noniid_alpha_0_5": {
            "dataset": {
                "name": "mnist",
                "root": "data",
                "partition_mode": "dirichlet",
                "num_clients": 10,
                "dirichlet_alpha": 0.5,
                "download": True,
                "num_classes": 10,
                "max_train_samples": None,
                "max_test_samples": None,
            },
            "model": {"name": "mnist_cnn"},
        },
        "mnist_noniid_alpha_0_1": {
            "dataset": {
                "name": "mnist",
                "root": "data",
                "partition_mode": "dirichlet",
                "num_clients": 10,
                "dirichlet_alpha": 0.1,
                "download": True,
                "num_classes": 10,
                "max_train_samples": None,
                "max_test_samples": None,
            },
            "model": {"name": "mnist_cnn"},
        },
        "cifar10_iid": {
            "dataset": {
                "name": "cifar10",
                "root": "data",
                "partition_mode": "iid",
                "num_clients": 10,
                "dirichlet_alpha": 0.1,
                "download": True,
                "num_classes": 10,
                "max_train_samples": None,
                "max_test_samples": None,
            },
            "model": {"name": "cifar_cnn"},
        },
        "cifar10_noniid_alpha_1_0": {
            "dataset": {
                "name": "cifar10",
                "root": "data",
                "partition_mode": "dirichlet",
                "num_clients": 10,
                "dirichlet_alpha": 1.0,
                "download": True,
                "num_classes": 10,
                "max_train_samples": None,
                "max_test_samples": None,
            },
            "model": {"name": "cifar_cnn"},
        },
        "cifar10_noniid_alpha_0_5": {
            "dataset": {
                "name": "cifar10",
                "root": "data",
                "partition_mode": "dirichlet",
                "num_clients": 10,
                "dirichlet_alpha": 0.5,
                "download": True,
                "num_classes": 10,
                "max_train_samples": None,
                "max_test_samples": None,
            },
            "model": {"name": "cifar_cnn"},
        },
        "cifar10_noniid_alpha_0_1": {
            "dataset": {
                "name": "cifar10",
                "root": "data",
                "partition_mode": "dirichlet",
                "num_clients": 10,
                "dirichlet_alpha": 0.1,
                "download": True,
                "num_classes": 10,
                "max_train_samples": None,
                "max_test_samples": None,
            },
            "model": {"name": "cifar_cnn"},
        },
    }
    for name, payload in dataset_defs.items():
        dump_yaml(FINAL_ROOT / "common" / "datasets" / f"{name}.yaml", payload)

    methods = {
        "fedavg": {
            "training": {"algorithm": "fedavg", "proximal_mu": 0.0},
            "privacy": {"dp_mode": "none", "epsilon": None, "noise_schedule": "none"},
            "compression": {"mode": "none", "topk_ratio": 1.0, "compress_updates": True},
        },
        "fedprox": {
            "training": {"algorithm": "fedprox", "proximal_mu": 0.5},
            "privacy": {"dp_mode": "none", "epsilon": None, "noise_schedule": "none"},
            "compression": {"mode": "none", "topk_ratio": 1.0, "compress_updates": True},
        },
        "fednova": {
            "training": {"algorithm": "fednova", "proximal_mu": 0.0},
            "privacy": {"dp_mode": "none", "epsilon": None, "noise_schedule": "none"},
            "compression": {"mode": "none", "topk_ratio": 1.0, "compress_updates": True},
        },
        "fedavg_fixed_dp": {
            "training": {"algorithm": "fedavg", "proximal_mu": 0.0},
            "privacy": {
                "dp_mode": "fixed",
                "epsilon": 650.0,
                "delta": 1.0e-5,
                "clip_norm": 1.0,
                "noise_multiplier": 0.12,
                "noise_schedule": "none",
                "accountant": "gaussian",
            },
            "compression": {"mode": "none", "topk_ratio": 1.0, "compress_updates": True},
        },
        "fedprox_fixed_dp": {
            "training": {"algorithm": "fedprox", "proximal_mu": 0.5},
            "privacy": {
                "dp_mode": "fixed",
                "epsilon": 650.0,
                "delta": 1.0e-5,
                "clip_norm": 1.0,
                "noise_multiplier": 0.12,
                "noise_schedule": "none",
                "accountant": "gaussian",
            },
            "compression": {"mode": "none", "topk_ratio": 1.0, "compress_updates": True},
        },
        "fednova_fixed_dp": {
            "training": {"algorithm": "fednova", "proximal_mu": 0.0},
            "privacy": {
                "dp_mode": "fixed",
                "epsilon": 650.0,
                "delta": 1.0e-5,
                "clip_norm": 1.0,
                "noise_multiplier": 0.12,
                "noise_schedule": "none",
                "accountant": "gaussian",
            },
            "compression": {"mode": "none", "topk_ratio": 1.0, "compress_updates": True},
        },
        "fedavg_adaptive_dp": {
            "training": {"algorithm": "fedavg", "proximal_mu": 0.0},
            "privacy": {
                "dp_mode": "adaptive",
                "epsilon": None,
                "target_epsilon": 650.0,
                "delta": 1.0e-5,
                "clip_norm": 1.0,
                "noise_multiplier": 0.10,
                "noise_schedule": "performance_budget",
                "min_noise_multiplier": 0.08,
                "max_noise_multiplier": 0.18,
                "schedule_metric": "test_accuracy",
                "schedule_warmup_rounds": 5,
                "adaptive_step": 0.01,
                "plateau_patience": 3,
                "schedule_metric_tolerance": 0.002,
                "schedule_drop_tolerance": 0.005,
                "budget_tolerance_ratio": 0.05,
                "client_aware_clipping": True,
                "client_clipping_metric": "update_norm",
                "client_clipping_beta": 0.5,
                "min_clip_norm": 0.6,
                "max_clip_norm": 1.4,
                "client_aware_noise": True,
                "client_noise_beta": 0.20,
                "reliability_aware_aggregation": True,
                "aggregation_reliability_beta": 0.4,
                "min_reliability_multiplier": 0.7,
                "max_reliability_multiplier": 1.0,
                "accountant": "gaussian",
            },
            "compression": {"mode": "none", "topk_ratio": 1.0, "compress_updates": True},
        },
        "fedprox_adaptive_dp": {
            "training": {"algorithm": "fedprox", "proximal_mu": 0.5},
            "privacy": {
                "dp_mode": "adaptive",
                "epsilon": None,
                "target_epsilon": 650.0,
                "delta": 1.0e-5,
                "clip_norm": 1.0,
                "noise_multiplier": 0.10,
                "noise_schedule": "performance_budget",
                "min_noise_multiplier": 0.08,
                "max_noise_multiplier": 0.18,
                "schedule_metric": "test_accuracy",
                "schedule_warmup_rounds": 5,
                "adaptive_step": 0.01,
                "plateau_patience": 3,
                "schedule_metric_tolerance": 0.002,
                "schedule_drop_tolerance": 0.005,
                "budget_tolerance_ratio": 0.05,
                "client_aware_clipping": True,
                "client_clipping_metric": "update_norm",
                "client_clipping_beta": 0.5,
                "min_clip_norm": 0.6,
                "max_clip_norm": 1.4,
                "client_aware_noise": True,
                "client_noise_beta": 0.20,
                "reliability_aware_aggregation": True,
                "aggregation_reliability_beta": 0.4,
                "min_reliability_multiplier": 0.7,
                "max_reliability_multiplier": 1.0,
                "accountant": "gaussian",
            },
            "compression": {"mode": "none", "topk_ratio": 1.0, "compress_updates": True},
        },
        "fednova_adaptive_dp": {
            "training": {"algorithm": "fednova", "proximal_mu": 0.0},
            "privacy": {
                "dp_mode": "adaptive",
                "epsilon": None,
                "target_epsilon": 650.0,
                "delta": 1.0e-5,
                "clip_norm": 1.0,
                "noise_multiplier": 0.10,
                "noise_schedule": "performance_budget",
                "min_noise_multiplier": 0.08,
                "max_noise_multiplier": 0.18,
                "schedule_metric": "test_accuracy",
                "schedule_warmup_rounds": 5,
                "adaptive_step": 0.01,
                "plateau_patience": 3,
                "schedule_metric_tolerance": 0.002,
                "schedule_drop_tolerance": 0.005,
                "budget_tolerance_ratio": 0.05,
                "client_aware_clipping": True,
                "client_clipping_metric": "update_norm",
                "client_clipping_beta": 0.5,
                "min_clip_norm": 0.6,
                "max_clip_norm": 1.4,
                "client_aware_noise": True,
                "client_noise_beta": 0.20,
                "reliability_aware_aggregation": True,
                "aggregation_reliability_beta": 0.4,
                "min_reliability_multiplier": 0.7,
                "max_reliability_multiplier": 1.0,
                "accountant": "gaussian",
            },
            "compression": {"mode": "none", "topk_ratio": 1.0, "compress_updates": True},
        },
    }
    for name, payload in methods.items():
        dump_yaml(FINAL_ROOT / "common" / "methods" / f"{name}.yaml", payload)

    dump_text(
        FINAL_ROOT / "README.md",
        (
            "# final configs\n\n"
            "本目录按最终实验组织配置。\n"
            "公共参数在 `common/` 下，具体实验配置在各实验子目录下。\n"
        ),
    )


def write_concrete_configs() -> None:
    experiments = {
        "exp_a_iid": {
            "conditions": [("mnist", "mnist_iid", "mnist_strong"), ("cifar10", "cifar10_iid", "cifar10_strong")],
            "output_root": "outputs/final/exp_a_iid",
        },
        "exp_a_noniid_alpha_0_1": {
            "conditions": [
                ("mnist", "mnist_noniid_alpha_0_1", "mnist_strong"),
                ("cifar10", "cifar10_noniid_alpha_0_1", "cifar10_strong"),
            ],
            "output_root": "outputs/final/exp_a_noniid_alpha_0_1",
        },
        "exp_b_alpha_sensitivity_alpha_1_0": {
            "conditions": [
                ("mnist", "mnist_noniid_alpha_1_0", "mnist_strong"),
                ("cifar10", "cifar10_noniid_alpha_1_0", "cifar10_strong"),
            ],
            "output_root": "outputs/final/exp_b_alpha_sensitivity/alpha_1_0",
        },
        "exp_b_alpha_sensitivity_alpha_0_5": {
            "conditions": [
                ("mnist", "mnist_noniid_alpha_0_5", "mnist_strong"),
                ("cifar10", "cifar10_noniid_alpha_0_5", "cifar10_strong"),
            ],
            "output_root": "outputs/final/exp_b_alpha_sensitivity/alpha_0_5",
        },
        "exp_b_alpha_sensitivity_alpha_0_1": {
            "conditions": [
                ("mnist", "mnist_noniid_alpha_0_1", "mnist_strong"),
                ("cifar10", "cifar10_noniid_alpha_0_1", "cifar10_strong"),
            ],
            "output_root": "outputs/final/exp_b_alpha_sensitivity/alpha_0_1",
        },
        "exp_d_privacy_tradeoff": {
            "conditions": [
                ("mnist", "mnist_noniid_alpha_0_1", "mnist_strong"),
                ("cifar10", "cifar10_noniid_alpha_0_1", "cifar10_strong"),
            ],
            "output_root": "outputs/final/exp_d_privacy_tradeoff",
        },
        "exp_e_final_summary": {
            "conditions": [
                ("mnist", "mnist_noniid_alpha_0_1", "mnist_strong"),
                ("cifar10", "cifar10_noniid_alpha_0_1", "cifar10_strong"),
            ],
            "output_root": "outputs/final/exp_e_final_summary",
        },
    }

    methods = [
        "fedavg",
        "fedprox",
        "fednova",
        "fedavg_fixed_dp",
        "fedprox_fixed_dp",
        "fednova_fixed_dp",
        "fedavg_adaptive_dp",
        "fedprox_adaptive_dp",
        "fednova_adaptive_dp",
    ]

    for experiment_name, spec in experiments.items():
        for dataset_name, dataset_config_name, training_config_name in spec["conditions"]:
            for method_name in methods:
                config_payload = {
                    "inherits": [
                        "../../common/runtime_ubuntu.yaml",
                        f"../../common/datasets/{dataset_config_name}.yaml",
                        f"../../common/training/{training_config_name}.yaml",
                        f"../../common/methods/{method_name}.yaml",
                    ],
                    "experiment": {
                        "name": f"{experiment_name}_{dataset_name}_{method_name}",
                        "seed": 42,
                        "output_root": spec["output_root"],
                    },
                }
                dump_yaml(
                    FINAL_ROOT / experiment_name / dataset_name / f"{method_name}.yaml",
                    config_payload,
                )


def archive_legacy_configs() -> None:
    LEGACY_ROOT.mkdir(parents=True, exist_ok=True)
    for path in sorted(EXPERIMENTS_ROOT.glob("*.yaml")):
        target = LEGACY_ROOT / path.name
        shutil.move(str(path), str(target))


def main() -> None:
    write_common_configs()
    write_concrete_configs()
    archive_legacy_configs()


if __name__ == "__main__":
    main()
