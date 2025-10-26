# -*- coding: utf-8 -*-
"""Bayesian safety gate leveraging Beta posterior confidence bounds."""

from __future__ import annotations

from typing import Dict, Optional

from ..bayes.posteriors import BetaPosterior


class BayesSafetyGate:
    """Maintain Beta posteriors per action category and return risk-aware decisions."""

    def __init__(
        self,
        *,
        categories: Optional[Dict[str, BetaPosterior]] = None,
        z: float = 1.64,
        read_only_th: float = 0.2,
        block_th: float = 0.4,
    ) -> None:
        self._posteriors: Dict[str, BetaPosterior] = categories or {}
        self.z = float(z)
        self.read_only_th = float(read_only_th)
        self.block_th = float(block_th)

    def ensure(self, name: str) -> None:
        if name not in self._posteriors:
            self._posteriors[name] = BetaPosterior(1.0, 1.0, half_life_tau=8.0)

    def decide(self, name: str, *, d_tau: float = 0.0) -> str:
        self.ensure(name)
        upper = self._posteriors[name].ci_upper(self.z)
        if upper >= self.block_th:
            return "BLOCK"
        if upper >= self.read_only_th:
            return "READ_ONLY"
        return "ALLOW"

    def risk_upper(self, name: str) -> float:
        self.ensure(name)
        return self._posteriors[name].ci_upper(self.z)

    def update(self, name: str, *, misfire: bool, d_tau: float) -> None:
        self.ensure(name)
        self._posteriors[name].update(success=(not misfire), d_tau=d_tau)

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        return {name: post.to_dict() for name, post in self._posteriors.items()}


__all__ = ["BayesSafetyGate"]
