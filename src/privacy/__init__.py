"""Differential privacy components."""

from src.privacy.accountant import PrivacyAccountant, estimate_epsilon_increment
from src.privacy.adaptive_dp import AdaptiveDPScheduler
from src.privacy.client_adaptive import (
    compute_client_adaptive_clip,
    compute_client_reliability_multiplier,
    compute_client_risk_boosted_noise,
)
from src.privacy.config import resolve_privacy_config
from src.privacy.mechanisms import clip_and_add_noise, compute_global_grad_norm
from src.privacy.mia import MembershipAttackResult, run_membership_inference_attack
from src.privacy.whitebox_attack import (
    GradientLeakageResult,
    collect_observed_gradients,
    evaluate_reconstruction,
    reconstruct_single_sample_from_gradients,
)

__all__ = [
    "AdaptiveDPScheduler",
    "PrivacyAccountant",
    "clip_and_add_noise",
    "compute_client_adaptive_clip",
    "compute_client_reliability_multiplier",
    "compute_client_risk_boosted_noise",
    "compute_global_grad_norm",
    "estimate_epsilon_increment",
    "MembershipAttackResult",
    "GradientLeakageResult",
    "collect_observed_gradients",
    "evaluate_reconstruction",
    "resolve_privacy_config",
    "reconstruct_single_sample_from_gradients",
    "run_membership_inference_attack",
]
