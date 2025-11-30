"""Life indicator dataclasses (identity / qualia / meta-awareness)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LifeIndicator:
    """Day-level snapshot of EQNet's wellbeing vector."""

    identity_score: float
    qualia_score: float
    meta_awareness_score: float

    def clamp(self) -> "LifeIndicator":
        """Clamp all scores into [0, 1] and return a new instance."""

        def _clip(value: float) -> float:
            return max(0.0, min(1.0, float(value)))

        return LifeIndicator(
            identity_score=_clip(self.identity_score),
            qualia_score=_clip(self.qualia_score),
            meta_awareness_score=_clip(self.meta_awareness_score),
        )
