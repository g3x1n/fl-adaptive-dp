"""Federated server loop for synchronous FedAvg, FedProx and FedNova training."""

from __future__ import annotations

import math
import random

from torch import nn

from src.fl.client import LocalClient
from src.fl.trainer import evaluate_model
from src.optim import aggregate_fedavg, aggregate_fednova


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

    def _sample_clients(self) -> list[LocalClient]:
        """Sample a fixed fraction of clients each round."""
        fraction_fit = self.config["training"]["fraction_fit"]
        num_selected = max(1, math.ceil(len(self.clients) * fraction_fit))
        return random.sample(self.clients, num_selected)

    def _estimate_payload_bytes(self) -> int:
        """Estimate upload size from the global model parameter tensor sizes."""
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.global_model.state_dict().values()
        )

    def run(self) -> dict[str, float]:
        """Execute synchronous federated rounds for the configured algorithm."""
        rounds = self.config["training"]["rounds"]
        local_epochs = self.config["training"]["local_epochs"]
        learning_rate = self.config["training"]["learning_rate"]
        algorithm = self.config["training"]["algorithm"].lower()
        proximal_mu = self.config["training"]["proximal_mu"]
        global_state = self.global_model.state_dict()

        last_eval = {"loss": 0.0, "accuracy": 0.0}

        for round_idx in range(1, rounds + 1):
            selected_clients = self._sample_clients()
            self.global_model.load_state_dict(global_state)

            client_states = []
            client_weights = []
            client_losses = []
            client_steps = []

            for client in selected_clients:
                # Every client starts from the same global parameters and trains locally.
                state, num_samples, avg_loss, local_steps = client.fit(
                    global_model=self.global_model,
                    local_epochs=local_epochs,
                    learning_rate=learning_rate,
                    algorithm=algorithm,
                    proximal_mu=proximal_mu,
                )
                client_states.append(state)
                client_weights.append(num_samples)
                client_losses.append(avg_loss)
                client_steps.append(local_steps)

            if algorithm in {"fedavg", "fedprox"}:
                # FedProx keeps FedAvg's server-side aggregation and only changes local loss.
                global_state = aggregate_fedavg(client_states, client_weights)
            elif algorithm == "fednova":
                # FedNova normalizes client updates before forming the global step.
                global_state = aggregate_fednova(
                    global_state=global_state,
                    client_states=client_states,
                    client_weights=client_weights,
                    client_steps=client_steps,
                )
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            self.global_model.load_state_dict(global_state)
            self.global_model.to(self.device)
            last_eval = evaluate_model(self.global_model, self.test_loader, self.device)

            train_loss = sum(client_losses) / max(len(client_losses), 1)
            payload = self._estimate_payload_bytes() * len(selected_clients)

            self.metrics_writer.write(
                {
                    "round": round_idx,
                    "train_loss": round(train_loss, 6),
                    "test_loss": round(last_eval["loss"], 6),
                    "test_accuracy": round(last_eval["accuracy"], 6),
                    "epsilon_spent": None,
                    "upload_payload_bytes": payload,
                    "selected_clients": len(selected_clients),
                    "algorithm": self.config["training"]["algorithm"],
                    "dp_mode": self.config["privacy"]["dp_mode"],
                    "compression_mode": self.config["compression"]["mode"],
                }
            )

            self.logger.info(
                "Round %s finished | algorithm=%s | train_loss=%.4f | test_loss=%.4f | test_accuracy=%.4f | selected_clients=%s",
                round_idx,
                self.config["training"]["algorithm"],
                train_loss,
                last_eval["loss"],
                last_eval["accuracy"],
                len(selected_clients),
            )

        return {
            "final_test_loss": last_eval["loss"],
            "final_test_accuracy": last_eval["accuracy"],
        }
