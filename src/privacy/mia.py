"""Simple black-box membership inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from src.utils.io import ensure_dir


@dataclass
class MembershipAttackResult:
    """Container for attack summary metrics."""

    auc_loss: float
    auc_confidence: float
    best_acc_loss: float
    best_acc_confidence: float
    member_mean_loss_score: float
    nonmember_mean_loss_score: float
    member_mean_confidence: float
    nonmember_mean_confidence: float


def sample_dataset_subset(dataset: Dataset, size: int, seed: int) -> Dataset:
    """Sample a deterministic subset for attack evaluation."""
    if size >= len(dataset):
        return dataset

    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    return Subset(dataset, indices[:size])


def _score_dataset(model, dataset: Dataset, batch_size: int, device) -> list[dict[str, float]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    scores: list[dict[str, float]] = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1)
            per_sample_loss = F.cross_entropy(logits, targets, reduction="none")
            true_confidence = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)

            for loss_value, confidence_value in zip(per_sample_loss, true_confidence):
                scores.append(
                    {
                        "loss_score": float(-loss_value.item()),
                        "confidence_score": float(confidence_value.item()),
                    }
                )

    return scores


def _pairwise_auc(member_scores: list[float], nonmember_scores: list[float]) -> float:
    """Compute AUC via pairwise ranking."""
    total_pairs = len(member_scores) * len(nonmember_scores)
    if total_pairs == 0:
        return 0.5

    wins = 0.0
    for member_score in member_scores:
        for nonmember_score in nonmember_scores:
            if member_score > nonmember_score:
                wins += 1.0
            elif math.isclose(member_score, nonmember_score):
                wins += 0.5
    return wins / total_pairs


def _best_threshold_accuracy(member_scores: list[float], nonmember_scores: list[float]) -> float:
    labels = [1] * len(member_scores) + [0] * len(nonmember_scores)
    scores = member_scores + nonmember_scores
    if not scores:
        return 0.0

    thresholds = sorted(set(scores))
    best_accuracy = 0.0
    for threshold in thresholds:
        predictions = [1 if score >= threshold else 0 for score in scores]
        correct = sum(int(pred == label) for pred, label in zip(predictions, labels))
        best_accuracy = max(best_accuracy, correct / len(labels))
    return best_accuracy


def run_membership_inference_attack(
    *,
    model,
    member_dataset: Dataset,
    nonmember_dataset: Dataset,
    batch_size: int,
    device,
    output_csv_path: str | Path | None = None,
) -> MembershipAttackResult:
    """Run a simple black-box MIA using loss and true-label confidence."""
    member_scores = _score_dataset(model, member_dataset, batch_size=batch_size, device=device)
    nonmember_scores = _score_dataset(model, nonmember_dataset, batch_size=batch_size, device=device)

    member_loss_scores = [item["loss_score"] for item in member_scores]
    nonmember_loss_scores = [item["loss_score"] for item in nonmember_scores]
    member_conf_scores = [item["confidence_score"] for item in member_scores]
    nonmember_conf_scores = [item["confidence_score"] for item in nonmember_scores]

    if output_csv_path is not None:
        output_path = Path(output_csv_path)
        ensure_dir(output_path.parent)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["membership", "loss_score", "confidence_score"],
            )
            writer.writeheader()
            for item in member_scores:
                writer.writerow({"membership": 1, **item})
            for item in nonmember_scores:
                writer.writerow({"membership": 0, **item})

    return MembershipAttackResult(
        auc_loss=_pairwise_auc(member_loss_scores, nonmember_loss_scores),
        auc_confidence=_pairwise_auc(member_conf_scores, nonmember_conf_scores),
        best_acc_loss=_best_threshold_accuracy(member_loss_scores, nonmember_loss_scores),
        best_acc_confidence=_best_threshold_accuracy(member_conf_scores, nonmember_conf_scores),
        member_mean_loss_score=sum(member_loss_scores) / max(len(member_loss_scores), 1),
        nonmember_mean_loss_score=sum(nonmember_loss_scores) / max(len(nonmember_loss_scores), 1),
        member_mean_confidence=sum(member_conf_scores) / max(len(member_conf_scores), 1),
        nonmember_mean_confidence=sum(nonmember_conf_scores) / max(len(nonmember_conf_scores), 1),
    )
