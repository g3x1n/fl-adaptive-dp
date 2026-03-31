"""Federated server loop for synchronous FedAvg, FedProx and FedNova training."""

from __future__ import annotations

import math
import random
import statistics

from torch import nn

from src.compression import apply_model_update
from src.fl.client import LocalClient
from src.fl.trainer import evaluate_model
from src.optim import aggregate_fednova_updates, aggregate_weighted_updates
from src.privacy import (
    AdaptiveDPScheduler,
    compute_client_adaptive_clip,
    compute_client_reliability_multiplier,
    compute_client_risk_boosted_noise,
    resolve_privacy_config,
)
from src.training import resolve_round_learning_rate


class SynchronousFLServer:
    """Coordinate client sampling, aggregation and global evaluation."""

    def __init__(
        self,
        global_model: nn.Module,
        clients: list[LocalClient],
        test_loader,
        device,
        metrics_writer,
        round_jsonl_writer,
        client_jsonl_writer,
        logger,
        config: dict,
    ) -> None:
        self.global_model = global_model
        self.clients = clients
        self.test_loader = test_loader
        self.device = device
        self.metrics_writer = metrics_writer
        self.round_jsonl_writer = round_jsonl_writer
        self.client_jsonl_writer = client_jsonl_writer
        self.logger = logger
        self.config = config
        self.privacy_config = resolve_privacy_config(config)
        self.adaptive_scheduler = AdaptiveDPScheduler(self.privacy_config)
        self.client_metric_history: dict[int, float] = {}

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

    def _build_client_privacy_plan(self, client_id: int, base_privacy_plan: dict) -> dict:
        """Adjust the shared privacy plan using client-level drift history."""
        if (
            not self.privacy_config.get("client_aware_clipping", False)
            and not self.privacy_config.get("client_aware_noise", False)
        ):
            return dict(base_privacy_plan)

        reference_metric = None
        if self.client_metric_history:
            reference_metric = statistics.median(self.client_metric_history.values())

        client_metric = self.client_metric_history.get(client_id)
        client_plan = dict(base_privacy_plan)
        reason = str(base_privacy_plan.get("schedule_reason", "adaptive plan"))
        client_num_samples = len(self.clients[client_id].dataloader.dataset)
        reference_num_samples = statistics.median(len(client.dataloader.dataset) for client in self.clients)
        reason_parts = [reason]

        if self.privacy_config.get("client_aware_clipping", False):
            adjusted_clip = compute_client_adaptive_clip(
                base_clip_norm=float(base_privacy_plan["clip_norm"]),
                client_metric=client_metric,
                reference_metric=reference_metric,
                beta=float(self.privacy_config.get("client_clipping_beta", 0.5)),
                min_clip_norm=float(self.privacy_config.get("min_clip_norm", base_privacy_plan["clip_norm"])),
                max_clip_norm=float(self.privacy_config.get("max_clip_norm", base_privacy_plan["clip_norm"])),
            )
            client_plan["clip_norm"] = adjusted_clip
            if client_metric is None or reference_metric is None:
                reason_parts.append("client-aware clip warmup")
            else:
                reason_parts.append(
                    f"client-aware clip prev_update_norm={client_metric:.4f} vs median={reference_metric:.4f}"
                )

        if self.privacy_config.get("client_aware_noise", False):
            adjusted_noise = compute_client_risk_boosted_noise(
                base_noise_multiplier=float(base_privacy_plan["noise_multiplier"]),
                client_metric=client_metric,
                reference_metric=reference_metric,
                client_num_samples=client_num_samples,
                reference_num_samples=reference_num_samples,
                beta=float(self.privacy_config.get("client_noise_beta", 0.25)),
                min_noise_multiplier=float(self.privacy_config.get("min_noise_multiplier", base_privacy_plan["noise_multiplier"])),
                max_noise_multiplier=float(self.privacy_config.get("max_noise_multiplier", base_privacy_plan["noise_multiplier"])),
            )
            client_plan["noise_multiplier"] = adjusted_noise
            reason_parts.append(
                f"client-aware noise samples={client_num_samples} median_samples={reference_num_samples:.1f}"
            )

        client_plan["schedule_reason"] = "; ".join(reason_parts)
        return client_plan

    def run(self) -> dict[str, float]:
        """Execute synchronous federated rounds for the configured algorithm."""
        rounds = self.config["training"]["rounds"]
        local_epochs = self.config["training"]["local_epochs"]
        algorithm = self.config["training"]["algorithm"].lower()
        proximal_mu = self.config["training"]["proximal_mu"]
        global_state = self.global_model.state_dict()

        last_eval = {"loss": 0.0, "accuracy": 0.0}
        best_test_accuracy = 0.0
        previous_round_metrics = {
            "train_loss": 0.0,
            "test_loss": 0.0,
            "test_accuracy": 0.0,
            "update_norm": 0.0,
            "epsilon_spent": 0.0,
        }
        round_noise_values: list[float] = []
        round_epsilons: list[float] = []

        for round_idx in range(1, rounds + 1):
            learning_rate = resolve_round_learning_rate(self.config["training"], round_idx)
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
            client_clip_norms = []
            client_noise_values = []
            client_ids = []

            for client in selected_clients:
                client_privacy_plan = self._build_client_privacy_plan(
                    client_id=client.client_id,
                    base_privacy_plan=privacy_plan,
                )
                # Every client starts from the same global parameters and trains locally.
                result = client.fit(
                    global_model=self.global_model,
                    local_epochs=local_epochs,
                    learning_rate=learning_rate,
                    algorithm=algorithm,
                    proximal_mu=proximal_mu,
                    optimizer_config={
                        "momentum": self.config["training"].get("optimizer_momentum", 0.0),
                        "weight_decay": self.config["training"].get("weight_decay", 0.0),
                        "nesterov": self.config["training"].get("nesterov", False),
                    },
                    privacy_plan=client_privacy_plan,
                    compression_config={
                        **self.config["compression"],
                        "batch_size": self.config["training"]["batch_size"],
                    },
                )
                client_updates.append(result["update"])
                client_ids.append(client.client_id)
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
                client_clip_norms.append(float(result["clip_norm"]))
                client_noise_values.append(float(result["noise_multiplier"]))
                self.client_metric_history[client.client_id] = float(result["update_norm"])

            aggregation_weights = list(client_weights)
            reliability_multipliers = [1.0 for _ in client_weights]
            if self.privacy_config.get("reliability_aware_aggregation", False) and client_updates:
                median_update_norm = statistics.median(client_update_norms) if client_update_norms else 0.0
                median_noise = statistics.median(client_noise_values) if client_noise_values else 0.0
                reliability_multipliers = []
                for update_norm, noise_multiplier in zip(client_update_norms, client_noise_values):
                    reliability = compute_client_reliability_multiplier(
                        client_metric=update_norm,
                        reference_metric=median_update_norm,
                        client_noise_multiplier=noise_multiplier,
                        reference_noise_multiplier=median_noise,
                        beta=float(self.privacy_config.get("aggregation_reliability_beta", 0.4)),
                        min_multiplier=float(self.privacy_config.get("min_reliability_multiplier", 0.6)),
                        max_multiplier=float(self.privacy_config.get("max_reliability_multiplier", 1.0)),
                    )
                    reliability_multipliers.append(reliability)
                aggregation_weights = [
                    max(1e-6, float(weight) * multiplier)
                    for weight, multiplier in zip(client_weights, reliability_multipliers)
                ]

            if algorithm in {"fedavg", "fedprox"}:
                aggregated_update = aggregate_weighted_updates(client_updates, aggregation_weights)
            elif algorithm == "fednova":
                aggregated_update = aggregate_fednova_updates(
                    global_state=global_state,
                    client_updates=client_updates,
                    client_weights=aggregation_weights,
                    client_steps=client_steps,
                )
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            global_state = apply_model_update(global_state=global_state, update=aggregated_update)
            self.global_model.load_state_dict(global_state)
            self.global_model.to(self.device)
            last_eval = evaluate_model(self.global_model, self.test_loader, self.device)
            best_test_accuracy = max(best_test_accuracy, float(last_eval["accuracy"]))

            train_loss = sum(client_losses) / max(len(client_losses), 1)
            epsilon_spent = max(client_epsilons) if client_epsilons else 0.0
            payload = sum(upload_payloads)
            pre_payload = sum(pre_compression_payloads)
            avg_compression_ratio = sum(compression_ratios) / max(len(compression_ratios), 1)
            avg_nnz_params = sum(nnz_params) / max(len(nnz_params), 1)
            avg_update_norm = sum(client_update_norms) / max(len(client_update_norms), 1)
            avg_grad_norm = sum(grad_norms) / max(len(grad_norms), 1)
            avg_clip_norm = sum(client_clip_norms) / max(len(client_clip_norms), 1)
            median_update_norm = statistics.median(client_update_norms) if client_update_norms else 0.0
            max_update_norm = max(client_update_norms) if client_update_norms else 0.0
            median_grad_norm = statistics.median(grad_norms) if grad_norms else 0.0
            max_grad_norm = max(grad_norms) if grad_norms else 0.0
            median_client_loss = statistics.median(client_losses) if client_losses else 0.0
            max_client_loss = max(client_losses) if client_losses else 0.0
            median_client_epsilon = statistics.median(client_epsilons) if client_epsilons else 0.0
            max_client_epsilon = max(client_epsilons) if client_epsilons else 0.0
            min_noise = min(client_noise_values) if client_noise_values else 0.0
            max_noise = max(client_noise_values) if client_noise_values else 0.0
            median_noise = statistics.median(client_noise_values) if client_noise_values else 0.0
            min_clip = min(client_clip_norms) if client_clip_norms else 0.0
            max_clip = max(client_clip_norms) if client_clip_norms else 0.0
            median_clip = statistics.median(client_clip_norms) if client_clip_norms else 0.0

            self.metrics_writer.write(
                {
                    "round": round_idx,
                    "learning_rate": round(learning_rate, 6),
                    "train_loss": round(train_loss, 6),
                    "test_loss": round(last_eval["loss"], 6),
                    "test_accuracy": round(last_eval["accuracy"], 6),
                    "best_test_accuracy": round(best_test_accuracy, 6),
                    "epsilon_spent": round(epsilon_spent, 6),
                    "upload_payload_bytes": payload,
                    "pre_compression_payload_bytes": round(pre_payload, 2),
                    "selected_clients": len(selected_clients),
                    "algorithm": self.config["training"]["algorithm"],
                    "dp_mode": self.config["privacy"]["dp_mode"],
                    "compression_mode": self.config["compression"]["mode"],
                    "noise_multiplier": round(privacy_plan["noise_multiplier"], 6),
                    "noise_multiplier_min": round(min_noise, 6),
                    "noise_multiplier_max": round(max_noise, 6),
                    "noise_multiplier_median": round(median_noise, 6),
                    "clip_norm": round(avg_clip_norm, 6),
                    "clip_norm_min": round(min_clip, 6),
                    "clip_norm_max": round(max_clip, 6),
                    "clip_norm_median": round(median_clip, 6),
                    "schedule_reason": privacy_plan["schedule_reason"],
                    "compression_ratio": round(avg_compression_ratio, 6),
                    "nnz_params": round(avg_nnz_params, 2),
                    "avg_update_norm": round(avg_update_norm, 6),
                    "median_update_norm": round(median_update_norm, 6),
                    "max_update_norm": round(max_update_norm, 6),
                    "avg_grad_norm": round(avg_grad_norm, 6),
                    "median_grad_norm": round(median_grad_norm, 6),
                    "max_grad_norm": round(max_grad_norm, 6),
                    "median_client_loss": round(median_client_loss, 6),
                    "max_client_loss": round(max_client_loss, 6),
                    "median_client_epsilon": round(median_client_epsilon, 6),
                    "max_client_epsilon": round(max_client_epsilon, 6),
                }
            )

            if self.client_jsonl_writer is not None:
                for (
                    client_id,
                    num_samples,
                    client_loss,
                    local_steps,
                    client_epsilon,
                    update_norm,
                    grad_norm,
                    clip_norm,
                    noise_multiplier,
                    upload_payload,
                    pre_payload_client,
                    compression_ratio,
                    nnz_param,
                    reliability_multiplier,
                    aggregation_weight,
                ) in zip(
                    client_ids,
                    client_weights,
                    client_losses,
                    client_steps,
                    client_epsilons,
                    client_update_norms,
                    grad_norms,
                    client_clip_norms,
                    client_noise_values,
                    upload_payloads,
                    pre_compression_payloads,
                    compression_ratios,
                    nnz_params,
                    reliability_multipliers,
                    aggregation_weights,
                ):
                    self.client_jsonl_writer.write(
                        {
                            "round": round_idx,
                            "client_id": client_id,
                            "algorithm": self.config["training"]["algorithm"],
                            "dp_mode": self.config["privacy"]["dp_mode"],
                            "learning_rate": round(learning_rate, 6),
                            "num_samples": int(num_samples),
                            "client_loss": round(float(client_loss), 6),
                            "local_steps": int(local_steps),
                            "client_epsilon": round(float(client_epsilon), 6),
                            "update_norm": round(float(update_norm), 6),
                            "grad_norm": round(float(grad_norm), 6),
                            "clip_norm": round(float(clip_norm), 6),
                            "noise_multiplier": round(float(noise_multiplier), 6),
                            "upload_payload_bytes": round(float(upload_payload), 2),
                            "pre_compression_payload_bytes": round(float(pre_payload_client), 2),
                            "compression_ratio": round(float(compression_ratio), 6),
                            "nnz_params": round(float(nnz_param), 2),
                            "reliability_multiplier": round(float(reliability_multiplier), 6),
                            "aggregation_weight": round(float(aggregation_weight), 6),
                            "schedule_reason": str(privacy_plan["schedule_reason"]),
                        }
                    )

            if self.round_jsonl_writer is not None:
                self.round_jsonl_writer.write(
                    {
                        "round": round_idx,
                        "algorithm": self.config["training"]["algorithm"],
                        "dp_mode": self.config["privacy"]["dp_mode"],
                        "compression_mode": self.config["compression"]["mode"],
                        "learning_rate": round(learning_rate, 6),
                        "train_loss": round(train_loss, 6),
                        "test_loss": round(last_eval["loss"], 6),
                        "test_accuracy": round(last_eval["accuracy"], 6),
                        "best_test_accuracy": round(best_test_accuracy, 6),
                        "epsilon_spent": round(epsilon_spent, 6),
                        "selected_clients": len(selected_clients),
                        "payload_bytes": round(float(payload), 2),
                        "pre_compression_payload_bytes": round(float(pre_payload), 2),
                        "avg_compression_ratio": round(avg_compression_ratio, 6),
                        "avg_nnz_params": round(avg_nnz_params, 2),
                        "noise_multiplier": round(privacy_plan["noise_multiplier"], 6),
                        "noise_multiplier_min": round(min_noise, 6),
                        "noise_multiplier_max": round(max_noise, 6),
                        "noise_multiplier_median": round(median_noise, 6),
                        "clip_norm_avg": round(avg_clip_norm, 6),
                        "clip_norm_min": round(min_clip, 6),
                        "clip_norm_max": round(max_clip, 6),
                        "clip_norm_median": round(median_clip, 6),
                        "avg_update_norm": round(avg_update_norm, 6),
                        "median_update_norm": round(median_update_norm, 6),
                        "max_update_norm": round(max_update_norm, 6),
                        "avg_grad_norm": round(avg_grad_norm, 6),
                        "median_grad_norm": round(median_grad_norm, 6),
                        "max_grad_norm": round(max_grad_norm, 6),
                        "median_client_loss": round(median_client_loss, 6),
                        "max_client_loss": round(max_client_loss, 6),
                        "median_client_epsilon": round(median_client_epsilon, 6),
                        "max_client_epsilon": round(max_client_epsilon, 6),
                        "schedule_reason": str(privacy_plan["schedule_reason"]),
                    }
                )

            self.logger.info(
                "Round %s finished | algorithm=%s | lr=%.5f | train_loss=%.4f | test_loss=%.4f | test_accuracy=%.4f | epsilon=%.4f | noise=%.4f | compression_ratio=%.4f | selected_clients=%s",
                round_idx,
                self.config["training"]["algorithm"],
                learning_rate,
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
                "epsilon_spent": epsilon_spent,
            }
            round_noise_values.append(privacy_plan["noise_multiplier"])
            round_epsilons.append(epsilon_spent)

        return {
            "final_test_loss": last_eval["loss"],
            "final_test_accuracy": last_eval["accuracy"],
            "best_test_accuracy": best_test_accuracy,
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
