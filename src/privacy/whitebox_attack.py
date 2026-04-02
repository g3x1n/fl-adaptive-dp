"""Server-side white-box gradient inversion attacks."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from src.privacy.mechanisms import clip_and_add_noise


@dataclass
class GradientLeakageResult:
    """Summary of a white-box gradient inversion run."""

    recovered_label: int
    label_match: bool
    reconstruction_mse: float
    reconstruction_psnr: float
    final_gradient_loss: float


def collect_observed_gradients(
    *,
    model,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    dp_mode: str,
    clip_norm: float,
    noise_multiplier: float,
    noise_seed: int = 42,
) -> list[torch.Tensor]:
    """Compute the client-observed gradients for one local optimization step."""
    working_model = model
    working_model.zero_grad(set_to_none=True)
    logits = working_model(inputs)
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    if dp_mode in {"fixed", "adaptive"}:
        cpu_state = torch.random.get_rng_state()
        torch.manual_seed(noise_seed)
        clip_and_add_noise(
            model=working_model,
            clip_norm=clip_norm,
            noise_multiplier=noise_multiplier,
            effective_batch_size=inputs.size(0),
        )
        torch.random.set_rng_state(cpu_state)

    gradients = []
    for parameter in working_model.parameters():
        gradients.append(parameter.grad.detach().clone())
    working_model.zero_grad(set_to_none=True)
    return gradients


def infer_label_from_gradients(observed_gradients: list[torch.Tensor]) -> int:
    """Infer the single-sample label using the iDLG heuristic."""
    last_bias_grad = None
    for grad in reversed(observed_gradients):
        if grad.ndim == 1:
            last_bias_grad = grad
            break
    if last_bias_grad is None:
        raise ValueError("Could not find a bias gradient for iDLG label inference.")
    return int(torch.argmin(last_bias_grad).item())


def _total_variation(image: torch.Tensor) -> torch.Tensor:
    horizontal = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]))
    vertical = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return horizontal + vertical


def reconstruct_single_sample_from_gradients(
    *,
    model,
    observed_gradients: list[torch.Tensor],
    input_shape: tuple[int, ...],
    device: torch.device,
    steps: int = 300,
    learning_rate: float = 0.1,
    tv_weight: float = 1e-4,
    clamp_min: float = -1.0,
    clamp_max: float = 3.0,
) -> tuple[torch.Tensor, int, float]:
    """Reconstruct a single input sample from observed gradients."""
    recovered_label = infer_label_from_gradients(observed_gradients)
    target = torch.tensor([recovered_label], device=device)
    dummy_input = torch.randn((1, *input_shape), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([dummy_input], lr=learning_rate)

    final_loss_value = 0.0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        logits = model(dummy_input)
        loss = F.cross_entropy(logits, target)
        dummy_gradients = torch.autograd.grad(loss, tuple(model.parameters()), create_graph=True)
        grad_match_loss = torch.zeros(1, device=device)
        for dummy_grad, observed_grad in zip(dummy_gradients, observed_gradients):
            grad_match_loss = grad_match_loss + F.mse_loss(dummy_grad, observed_grad.to(device))
        total_loss = grad_match_loss + tv_weight * _total_variation(dummy_input)
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            dummy_input.clamp_(clamp_min, clamp_max)
        final_loss_value = float(grad_match_loss.item())

    return dummy_input.detach().cpu(), recovered_label, final_loss_value


def evaluate_reconstruction(
    *,
    original_input: torch.Tensor,
    original_label: int,
    reconstructed_input: torch.Tensor,
    recovered_label: int,
    final_gradient_loss: float,
) -> GradientLeakageResult:
    """Score the reconstruction quality."""
    mse = torch.mean((original_input.cpu() - reconstructed_input.cpu()) ** 2).item()
    psnr = 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
    return GradientLeakageResult(
        recovered_label=recovered_label,
        label_match=(int(original_label) == int(recovered_label)),
        reconstruction_mse=float(mse),
        reconstruction_psnr=float(psnr),
        final_gradient_loss=float(final_gradient_loss),
    )
