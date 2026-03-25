"""Federated server loop for synchronous FedAvg, FedProx and FedNova training."""

from __future__ import annotations

import math
import random

from torch import nn

from src.compression import apply_model_update
from src.fl.client import LocalClient
from src.fl.trainer import evaluate_model
from src.optim import aggregate_fednova_updates, aggregate_weighted_updates
from src.privacy import AdaptiveDPScheduler, resolve_privacy_config


class SynchronousFLServer:
    """Coordinate client sampling, aggregation and global evaluation."""

    def __init__(
        self,
        global_model: nn.Module,
        clients: list[LocalClient],
        test_loader,
        device,
        metrics_writer,
        logger,
        config: dict,
    ) -> None:
        self.global_model = global_model
        self.clients = clients
        self.test_loader = test_loader
        self.device = device
        self.metrics_writer = metrics_writer
        self.logger = logger
        self.config = config
        self.privacy_config = resolve_privacy_config(config)
        self.adaptive_scheduler = AdaptiveDPScheduler(self.privacy_config)

    def _sample_clients(self) -> list[LocalClient]:
        """Sample a fixed fraction of clients each round."""
        fraction_fit = self.config["training"]["fraction_fit"]
        num_selected = max(1, math.ceil(len(self.clients) * fraction_fit))
        return random.sample(self.clients, num_selected)

    def _build_privacy_plan(
        self,
        round_idx: int,
        total_rounds: int,
        previous_round_metrics: dict[str, float],
    ) -> dict:
        """Create the per-round privacy plan used by all selected clients."""
        mode = self.privacy_config["dp_mode"]
        if mode == "none":
            return {
                "dp_mode": "none",
                "clip_norm": float(self.privacy_config["clip_norm"]),
                "noise_multiplier": 0.0,
                "delta": float(self.privacy_config["delta"]),
                "schedule_reason": "dp disabled",
            }

        if mode == "fixed":
            return {
                "dp_mode": "fixed",
                "clip_norm": float(self.privacy_config["clip_norm"]),
                "noise_multiplier": float(self.privacy_config["noise_multiplier"]),
                "delta": float(self.privacy_config["delta"]),
                "schedule_reason": "fixed schedule",
            }

        if mode == "adaptive":
            adaptive_plan = self.adaptive_scheduler.schedule(
                round_idx=round_idx,
                total_rounds=total_rounds,
                previous_metrics=previous_round_metrics,
            )
            adaptive_plan["delta"] = float(self.privacy_config["delta"])
            return adaptive_plan

        raise ValueError(f"Unsupported dp_mode: {mode}")

    def run(self) -> dict[str, float]:
        """Execute synchronous federated rounds for the configured algorithm."""
        rounds = self.config["training"]["rounds"]
        local_epochs = self.config["training"]["local_epochs"]
        learning_rate = self.config["training"]["learning_rate"]
        algorithm = self.config["training"]["algorithm"].lower()
        proximal_mu = self.config["training"]["proximal_mu"]
        global_state = self.global_model.state_dict()

        last_eval = {"loss": 0.0, "accuracy": 0.0}
        previous_round_metrics = {
            "train_loss": 0.0,
            "test_loss": 0.0,
            "test_accuracy": 0.0,
            "update_norm": 0.0,
        }
        round_noise_values: list[float] = []
        round_epsilons: list[float] = []

        for round_idx in range(1, rounds + 1):
            selected_clients = self._sample_clients()
            self.global_model.load_state_dict(global_state)
            privacy_plan = self._build_privacy_plan(
                round_idx=round_idx,
                total_rounds=rounds,
                previous_round_metrics=previous_round_metrics,
            )

            client_updates = []
            client_weights = []
            client_losses = []
            client_steps = []
            client_epsilons = []
            client_update_norms = []
            upload_payloads = []
            pre_compression_payloads = []
            compression_ratios = []
            nnz_params = []
            grad_norms = []

            for client in selected_clients:
                # Every client starts from the same global parameters and trains locally.
                result = client.fit(
                    global_model=self.global_model,
                    local_epochs=local_epochs,
                    learning_rate=learning_rate,
                    algorithm=algorithm,
                    proximal_mu=proximal_mu,
                    privacy_plan=privacy_plan,
                    compression_config={
                        **self.config["compression"],
                        "batch_size": self.config["training"]["batch_size"],
                    },
                )
                client_updates.append(result["update"])
                client_weights.append(result["num_samples"])
                client_losses.append(result["avg_loss"])
                client_steps.append(result["local_steps"])
                client_epsilons.append(result["epsilon_spent"])
                client_update_norms.append(result["update_norm"])
                upload_payloads.append(result["upload_payload_bytes"])
                pre_compression_payloads.append(result["pre_compression_payload_bytes"])
                compression_ratios.append(result["compression_ratio"])
                nnz_params.append(result["nnz_params"])
                grad_norms.append(result["avg_post_clip_grad_norm"])

            if algorithm in {"fedavg", "fedprox"}:
                aggregated_update = aggregate_weighted_updates(client_updates, client_weights)
            elif algorithm == "fednova":
                aggregated_update = aggregate_fednova_updates(
                    global_state=global_state,
                    client_updates=client_updates,
                    client_weights=client_weights,
                    client_steps=client_steps,
                )
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            global_state = apply_model_update(global_state=global_state, update=aggregated_update)
            self.global_model.load_state_dict(global_state)
            self.global_model.to(self.device)
            last_eval = evaluate_model(self.global_model, self.test_loader, self.device)

            train_loss = sum(client_losses) / max(len(client_losses), 1)
            epsilon_spent = max(client_epsilons) if client_epsilons else 0.0
            payload = sum(upload_payloads)
            pre_payload = sum(pre_compression_payloads)
            avg_compression_ratio = sum(compression_ratios) / max(len(compression_ratios), 1)
            avg_nnz_params = sum(nnz_params) / max(len(nnz_params), 1)
            avg_update_norm = sum(client_update_norms) / max(len(client_update_norms), 1)
            avg_grad_norm = sum(grad_norms) / max(len(grad_norms), 1)

            self.metrics_writer.write(
                {
                    "round": round_idx,
                    "train_loss": round(train_loss, 6),
                    "test_loss": round(last_eval["loss"], 6),
                    "test_accuracy": round(last_eval["accuracy"], 6),
                    "epsilon_spent": round(epsilon_spent, 6),
                    "upload_payload_bytes": payload,
                    "pre_compression_payload_bytes": round(pre_payload, 2),
                    "selected_clients": len(selected_clients),
                    "algorithm": self.config["training"]["algorithm"],
                    "dp_mode": self.config["privacy"]["dp_mode"],
                    "compression_mode": self.config["compression"]["mode"],
                    "noise_multiplier": round(privacy_plan["noise_multiplier"], 6),
                    "clip_norm": round(privacy_plan["clip_norm"], 6),
                    "schedule_reason": privacy_plan["schedule_reason"],
                    "compression_ratio": round(avg_compression_ratio, 6),
                    "nnz_params": round(avg_nnz_params, 2),
                    "avg_update_norm": round(avg_update_norm, 6),
                    "avg_grad_norm": round(avg_grad_norm, 6),
                }
            )

            self.logger.info(
                "Round %s finished | algorithm=%s | train_loss=%.4f | test_loss=%.4f | test_accuracy=%.4f | epsilon=%.4f | noise=%.4f | compression_ratio=%.4f | selected_clients=%s",
                round_idx,
                self.config["training"]["algorithm"],
                train_loss,
                last_eval["loss"],
                last_eval["accuracy"],
                epsilon_spent,
                privacy_plan["noise_multiplier"],
                avg_compression_ratio,
                len(selected_clients),
            )
            previous_round_metrics = {
                "train_loss": train_loss,
                "test_loss": last_eval["loss"],
                "test_accuracy": last_eval["accuracy"],
                "update_norm": avg_update_norm,
            }
            round_noise_values.append(privacy_plan["noise_multiplier"])
            round_epsilons.append(epsilon_spent)

        return {
            "final_test_loss": last_eval["loss"],
            "final_test_accuracy": last_eval["accuracy"],
            "privacy_summary": {
                "dp_mode": self.config["privacy"]["dp_mode"],
                "noise_schedule": self.config["privacy"]["noise_schedule"],
                "final_epsilon_spent": round(round_epsilons[-1], 6) if round_epsilons else 0.0,
                "min_noise_multiplier": round(min(round_noise_values), 6) if round_noise_values else 0.0,
                "max_noise_multiplier": round(max(round_noise_values), 6) if round_noise_values else 0.0,
            },
            "compression_summary": {
                "mode": self.config["compression"]["mode"],
            },
        }
