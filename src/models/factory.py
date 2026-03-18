"""Model factory for experiment configs."""

from __future__ import annotations

from torch import nn

from src.models.cifar_cnn import CIFAR10CNN
from src.models.mlp import MLPClassifier
from src.models.mnist_cnn import MNISTCNN


def build_model(model_name: str, dataset_name: str, num_classes: int) -> nn.Module:
    """Instantiate a supported baseline model from config values."""
    normalized_model = model_name.lower()
    normalized_dataset = dataset_name.lower()

    if normalized_model == "mnist_cnn":
        return MNISTCNN(num_classes=num_classes)

    if normalized_model == "cifar_cnn":
        return CIFAR10CNN(num_classes=num_classes)

    if normalized_model == "mlp":
        input_dim = 28 * 28 if normalized_dataset == "mnist" else 32 * 32 * 3
        return MLPClassifier(input_dim=input_dim, num_classes=num_classes)

    raise ValueError(f"Unsupported model: {model_name}")

