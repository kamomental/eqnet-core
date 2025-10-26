# -*- coding: utf-8 -*-
"""Self-as-distribution utilities (roles & narrative coherence)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import math


@dataclass
class RolePosterior:
    """Maintain role probabilities instead of a single fixed self."""

    roles: Sequence[str]
    temperature: float = 1.0
    halflife_tau: float = 12.0

    def __post_init__(self) -> None:
        if not self.roles:
            self.roles = ["caregiver", "playful", "researcher"]
        self.logits: Dict[str, float] = {role: 0.0 for role in self.roles}

    def decay(self, d_tau: float) -> None:
        lam = math.exp(-max(d_tau, 0.0) / max(self.halflife_tau, 1e-6))
        for role in self.roles:
            self.logits[role] *= lam

    def nudge(self, evidence: Dict[str, float] | None) -> None:
        if not evidence:
            return
        for role, delta in evidence.items():
            if role in self.logits:
                self.logits[role] += float(delta)

    def posterior(self) -> Dict[str, float]:
        if not self.logits:
            return {}
        scale = max(self.temperature, 1e-6)
        xs = [self.logits[r] / scale for r in self.roles]
        m = max(xs)
        exps = [math.exp(x - m) for x in xs]
        z = sum(exps) or 1.0
        probs = [val / z for val in exps]
        return {role: prob for role, prob in zip(self.roles, probs)}


@dataclass
class NarrativePosterior:
    """Track narrative consistency (supports vs contradicts) in [0, 1]."""

    supports: float = 0.0
    contradicts: float = 0.0

    def update(self, events: Iterable[Tuple[str, str]]) -> None:
        for kind, _ in events:
            tag = str(kind).lower()
            if tag == "supports":
                self.supports += 1.0
            elif tag == "contradicts":
                self.contradicts += 1.0

    def coherence(self) -> float:
        total = self.supports + self.contradicts
        if total <= 0.0:
            return 1.0
        return float(max(0.0, min(1.0, self.supports / total)))


def kld(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-6) -> float:
    """Compute KL(p || q) with tiny smoothing."""

    if not p or not q:
        return 0.0
    roles = set(p) | set(q)
    total = 0.0
    for role in roles:
        pv = max(eps, float(p.get(role, 0.0)))
        qv = max(eps, float(q.get(role, 0.0)))
        total += pv * math.log(pv / qv)
    return float(max(0.0, total))


__all__ = ["RolePosterior", "NarrativePosterior", "kld"]
