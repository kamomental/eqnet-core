# -*- coding: utf-8 -*-
"""Lightweight Bayesian posterior utilities with subjective-time decay."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Dict


def _decay_factor(d_tau: float, half_life_tau: float) -> float:
    if half_life_tau <= 0:
        return 1.0
    return math.exp(-math.log(2.0) * (d_tau / max(half_life_tau, 1e-9)))


@dataclass
class BetaPosterior:
    a: float = 1.0
    b: float = 1.0
    half_life_tau: float = 4.0

    def mean(self) -> float:
        return self.a / (self.a + self.b)

    def var(self) -> float:
        s = self.a + self.b
        return (self.a * self.b) / (s * s * (s + 1.0))

    def sample(self) -> float:
        x = random.gammavariate(self.a, 1.0)
        y = random.gammavariate(self.b, 1.0)
        return x / (x + y) if (x + y) > 0.0 else self.mean()

    def update(self, success: bool, *, d_tau: float = 0.0) -> None:
        lam = _decay_factor(d_tau, self.half_life_tau)
        self.a = 1.0 + lam * (self.a - 1.0)
        self.b = 1.0 + lam * (self.b - 1.0)
        if success:
            self.a += 1.0
        else:
            self.b += 1.0

    def ci_lower(self, z: float = 1.64) -> float:
        m, v = self.mean(), self.var()
        return max(0.0, min(1.0, m - z * math.sqrt(v)))

    def ci_upper(self, z: float = 1.64) -> float:
        m, v = self.mean(), self.var()
        return max(0.0, min(1.0, m + z * math.sqrt(v)))

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class DirichletPosterior:
    alpha: Dict[str, float]
    half_life_tau: float = 12.0

    def mean(self) -> Dict[str, float]:
        total = sum(self.alpha.values())
        if total <= 0.0:
            return {k: 1.0 / len(self.alpha) for k in self.alpha}
        return {k: v / total for k, v in self.alpha.items()}

    def update(self, counts: Dict[str, float], *, d_tau: float = 0.0) -> None:
        lam = _decay_factor(d_tau, self.half_life_tau)
        for key in list(self.alpha.keys()):
            base = self.alpha[key]
            self.alpha[key] = 1.0 + lam * (base - 1.0) + max(0.0, float(counts.get(key, 0.0)))

    def to_dict(self) -> Dict[str, object]:
        return {"alpha": dict(self.alpha), "half_life_tau": self.half_life_tau}


__all__ = ["BetaPosterior", "DirichletPosterior", "_decay_factor"]
