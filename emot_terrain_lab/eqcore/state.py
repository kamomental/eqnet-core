from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class Affect:
    """Low-dimensional affect estimate parsed from user input."""

    valence: float = 0.0
    arousal: float = 0.0
    care: float = 0.0
    novelty: float = 0.0

    def clamp(self, min_val: float = -1.0, max_val: float = 1.0) -> "Affect":
        return Affect(
            valence=float(max(min(self.valence, max_val), min_val)),
            arousal=float(max(min(self.arousal, max_val), min_val)),
            care=float(max(min(self.care, max_val), min_val)),
            novelty=float(max(min(self.novelty, max_val), min_val)),
        )


@dataclass
class EmotionState:
    """Slow-moving emotional mood."""

    valence: float = 0.0
    arousal: float = 0.0
    persistence: float = 0.5
    acceptance: float = 0.7

    def update(self, affect: Affect, alpha: float = 0.2) -> "EmotionState":
        return EmotionState(
            valence=(1 - alpha) * self.valence + alpha * affect.valence,
            arousal=(1 - alpha) * self.arousal + alpha * affect.arousal,
            persistence=self.persistence,
            acceptance=(1 - alpha) * self.acceptance + alpha * (0.5 + 0.5 * affect.care),
        )


@dataclass
class Stance:
    """Conversation stance used by the policy."""

    mode: str = "listen"  # listen | soften | guide
    confidence: float = 0.5


@dataclass
class Params:
    """Control parameters for Scalable-Softmax attention."""

    s: float = 0.6
    gamma: float = 0.6
    lam: float = 8.0

    s_min: float = 0.2
    s_max: float = 1.2
    gamma_min: float = 0.2
    gamma_max: float = 1.5
    lam_min: float = 4.0
    lam_max: float = 16.0

    def clamp(self) -> "Params":
        return Params(
            s=float(max(self.s_min, min(self.s, self.s_max))),
            gamma=float(max(self.gamma_min, min(self.gamma, self.gamma_max))),
            lam=float(max(self.lam_min, min(self.lam, self.lam_max))),
            s_min=self.s_min,
            s_max=self.s_max,
            gamma_min=self.gamma_min,
            gamma_max=self.gamma_max,
            lam_min=self.lam_min,
            lam_max=self.lam_max,
        )


@dataclass
class CoreState:
    """Persistent EQCore state."""

    Phi: torch.Tensor
    Psi: torch.Tensor
    rho: float
    R: float
    mood: EmotionState
    stance: Stance
    needs: Dict[str, float] = field(default_factory=dict)

    def to(self, device: torch.device) -> "CoreState":
        return CoreState(
            Phi=self.Phi.to(device),
            Psi=self.Psi.to(device),
            rho=self.rho,
            R=self.R,
            mood=self.mood,
            stance=self.stance,
            needs=self.needs.copy(),
        )


def initial_state(
    seq_len: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> CoreState:
    device = device or torch.device("cpu")
    Phi = torch.zeros(seq_len, dtype=dtype, device=device)
    Psi = torch.zeros(seq_len, dtype=dtype, device=device)
    return CoreState(
        Phi=Phi,
        Psi=Psi,
        rho=0.0,
        R=0.0,
        mood=EmotionState(),
        stance=Stance(),
        needs={},
    )
