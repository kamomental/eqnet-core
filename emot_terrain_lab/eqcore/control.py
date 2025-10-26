from __future__ import annotations

from dataclasses import replace
from typing import Tuple

import torch

from emot_terrain_lab.eqcore.state import Affect, CoreState, EmotionState, Params, Stance


def field_update(
    attn: torch.Tensor,
    state: CoreState,
    alpha: float = 0.35,
    beta: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Update field Psi and Phi using attention heatmap.

    Args:
        attn: attention probabilities [B, H, T, T]
        state: previous core state
        alpha: smoothing factor for Psi
        beta: smoothing factor for Phi

    Returns:
        (Psi_new, Phi_new) both [T]
    """
    attn_mean = attn.mean(dim=0)  # [H, T, T]
    diag_energy = attn_mean.mean(dim=0).diagonal()  # [T]
    diag_energy = diag_energy.clamp(0.0, 1.0)

    Psi_new = (1 - alpha) * state.Psi + alpha * diag_energy
    Phi_new = (1 - beta) * state.Phi + beta * Psi_new
    return Psi_new, Phi_new


def criticals(Phi: torch.Tensor, Psi: torch.Tensor, attn: torch.Tensor) -> Tuple[float, float]:
    """Compute rho (spectral norm proxy) and Kuramoto-like R."""
    rho = float(torch.linalg.vector_norm(Phi, ord=2))
    attn_mean = attn.mean(dim=0)  # [H, T, T]
    # treat attention rows as phases using soft angle mapping
    weights = attn_mean.mean(dim=0)  # [T, T]
    phase = torch.linspace(0, 2 * torch.pi, weights.size(-1), device=weights.device)
    complex_phase = torch.exp(1j * phase)
    center = torch.matmul(weights, complex_phase)
    R = float(torch.abs(center.mean()) / max(1e-6, torch.abs(complex_phase).mean()))
    return rho, max(0.0, min(R, 1.0))


def decide_stance(aff: Affect, rho: float, R: float, prev: Stance) -> Stance:
    """
    Choose stance based on affect and critical indicators.
    """
    arousal = aff.arousal
    novelty = aff.novelty

    if rho > 2.5 or arousal > 0.65:
        mode = "soften"
    elif R < 0.3 and novelty > 0.2:
        mode = "guide"
    else:
        mode = "listen"

    conf = max(0.1, 1.0 - abs(arousal - 0.4))
    if mode == prev.mode:
        conf = min(1.0, prev.confidence + 0.1)
    return Stance(mode=mode, confidence=conf)


def update_mood(mood: EmotionState, aff: Affect) -> EmotionState:
    return mood.update(aff)


def compute_entropy(attn: torch.Tensor) -> float:
    p = attn.clamp_min(1e-9)
    entropy = - (p * p.log()).sum(dim=-1).mean()
    return float(entropy)


def policy_update(
    params: Params,
    entropy_now: float,
    entropy_target: Tuple[float, float],
    rho: float,
    stance: Stance,
    eta_entropy: float = 0.02,
    eta_rho: float = 0.01,
) -> Params:
    """PID-lite update of attention control parameters."""
    H_min, H_max = entropy_target
    H_star = (H_min + H_max) / 2.0
    delta = H_star - entropy_now

    s_new = params.s + eta_entropy * delta
    gamma_new = params.gamma

    if stance.mode == "soften":
        gamma_new -= eta_rho * (rho - 1.0)
    elif stance.mode == "guide":
        gamma_new += eta_rho * (1.0 - rho)

    updated = Params(s=s_new, gamma=gamma_new, lam=params.lam)
    return updated.clamp()
