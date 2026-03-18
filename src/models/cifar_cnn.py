"""Baseline CNN for CIFAR-10 experiments."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class CIFAR10CNN(nn.Module):
    """A simple CNN suitable for small-scale CIFAR-10 federated runs."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs):
        x = self.features(inputs)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

