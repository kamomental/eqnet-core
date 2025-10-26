"""Minimal value committee stub."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ValueCommittee:
    """Evaluates actions across simple normative axes."""

    norms: Dict[str, float] = field(
        default_factory=lambda: {"care": 0.5, "fairness": 0.5, "harm": 0.5}
    )

    def vote(self, context: Dict[str, float]) -> Dict[str, float]:
        care = context.get("mood_v", 0.0)
        fairness = 1.0 - abs(context.get("self_vs_other", 0.0))
        harm = max(0.0, 0.5 - context.get("taste_score", 0.0))
        score = 0.4 * care + 0.4 * fairness - 0.2 * harm
        vote = {
            "score": max(-1.0, min(1.0, score)),
            "care": care,
            "fairness": fairness,
            "harm": harm,
        }
        return vote

    def update(self, feedback: Dict[str, float]) -> None:
        for k, v in feedback.items():
            if k in self.norms:
                self.norms[k] = 0.9 * self.norms[k] + 0.1 * v
