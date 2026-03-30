"""Client-aware privacy helpers for heterogeneous federated settings."""

from __future__ import annotations


def compute_client_adaptive_clip(
    *,
    base_clip_norm: float,
    client_metric: float | None,
    reference_metric: float | None,
    beta: float,
    min_clip_norm: float,
    max_clip_norm: float,
) -> float:
    """Adjust clip norm based on a client's previous-round update strength.

    Intuition:
    - clients with larger-than-typical updates are more drift-prone, so we
      tighten clipping for the next round;
    - clients with smaller, steadier updates can keep a looser clip norm.
    """
    if client_metric is None or reference_metric is None or reference_metric <= 0:
        return float(base_clip_norm)

    ratio = max(float(client_metric), 0.0) / max(float(reference_metric), 1e-8)
    if ratio > 1.0:
        adjusted = base_clip_norm / (1.0 + beta * (ratio - 1.0))
    else:
        adjusted = base_clip_norm * (1.0 + beta * (1.0 - ratio))

    return float(max(min_clip_norm, min(max_clip_norm, adjusted)))


def compute_client_risk_boosted_noise(
    *,
    base_noise_multiplier: float,
    client_metric: float | None,
    reference_metric: float | None,
    client_num_samples: int,
    reference_num_samples: float | None,
    beta: float,
    min_noise_multiplier: float,
    max_noise_multiplier: float,
) -> float:
    """Increase noise for high-risk clients without lowering protection elsewhere.

    A client is treated as higher risk when:
    - its previous update norm is larger than the typical client, or
    - it owns fewer local samples, which yields a higher effective sample rate.
    """
    adjusted = float(base_noise_multiplier)

    if client_metric is not None and reference_metric is not None and reference_metric > 0:
        metric_ratio = max(float(client_metric), 0.0) / max(float(reference_metric), 1e-8)
        if metric_ratio > 1.0:
            adjusted *= 1.0 + beta * (metric_ratio - 1.0)

    if reference_num_samples is not None and reference_num_samples > 0 and client_num_samples > 0:
        sample_ratio = float(reference_num_samples) / float(client_num_samples)
        if sample_ratio > 1.0:
            adjusted *= 1.0 + beta * (sample_ratio - 1.0)

    return float(max(min_noise_multiplier, min(max_noise_multiplier, adjusted)))


def compute_client_reliability_multiplier(
    *,
    client_metric: float | None,
    reference_metric: float | None,
    client_noise_multiplier: float,
    reference_noise_multiplier: float,
    beta: float,
    min_multiplier: float,
    max_multiplier: float,
) -> float:
    """Compute a trust multiplier for reliability-aware server aggregation.

    Clients with unusually large update norms or unusually high noise are
    downweighted slightly during aggregation to reduce noisy global steps.
    """
    if beta <= 0:
        return 1.0

    drift_penalty = 0.0
    if client_metric is not None and reference_metric is not None and reference_metric > 0:
        drift_ratio = max(float(client_metric), 0.0) / max(float(reference_metric), 1e-8)
        drift_penalty = max(0.0, drift_ratio - 1.0)

    noise_penalty = 0.0
    if reference_noise_multiplier > 0:
        noise_ratio = max(float(client_noise_multiplier), 0.0) / max(float(reference_noise_multiplier), 1e-8)
        noise_penalty = max(0.0, noise_ratio - 1.0)

    multiplier = 1.0 / (1.0 + beta * drift_penalty + beta * noise_penalty)
    return float(max(min_multiplier, min(max_multiplier, multiplier)))
