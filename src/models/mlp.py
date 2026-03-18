"""A small MLP baseline for flattened image inputs."""

from __future__ import annotations

import torch.nn as nn


class MLPClassifier(nn.Module):
    """A generic MLP that can be reused for simple baselines."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs):
        return self.network(inputs)

