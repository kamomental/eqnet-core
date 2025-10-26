# -*- coding: utf-8 -*-
"""Thompson-sampling persona selector."""

from __future__ import annotations

from typing import Dict, Iterable, List

from ..bayes.posteriors import BetaPosterior


class BayesPersonaSelector:
    """
    Treat persona modes as arms and select via Thompson sampling.
    """

    def __init__(self, modes: Iterable[str], *, half_life_tau: float = 6.0) -> None:
        self.modes: List[str] = list(modes)
        if not self.modes:
            raise ValueError("BayesPersonaSelector requires at least one mode.")
        self._posterior = {
            mode: BetaPosterior(1.0, 1.0, half_life_tau=half_life_tau) for mode in self.modes
        }

    def choose(self) -> str:
        scores = {mode: post.sample() for mode, post in self._posterior.items()}
        return max(scores.items(), key=lambda item: item[1])[0]

    def update(self, mode: str, *, success: bool, d_tau: float) -> None:
        if mode not in self._posterior:
            self._posterior[mode] = BetaPosterior(1.0, 1.0, half_life_tau=6.0)
        self._posterior[mode].update(success=success, d_tau=d_tau)

    def set_halflife_tau(self, half_life: float) -> None:
        """Update half-life for all tracked persona modes."""
        try:
            half = float(half_life)
        except Exception:
            return
        if half <= 0.0:
            return
        for post in self._posterior.values():
            post.half_life_tau = half

    def metrics(self) -> Dict[str, Dict[str, float]]:
        return {
            mode: {
                "mean": post.mean(),
                "ci_lower": post.ci_lower(),
                "ci_upper": post.ci_upper(),
                "a": post.a,
                "b": post.b,
            }
            for mode, post in self._posterior.items()
        }


__all__ = ["BayesPersonaSelector"]
