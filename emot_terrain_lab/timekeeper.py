# -*- coding: utf-8 -*-
"""Timekeeper for managing internal (subjective) time."""

from __future__ import annotations

import math
import time
from typing import Dict, Optional


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class TimeKeeper:
    """Track wall-clock dt and subjective d_tau for the agent.

    subjective rate is modulated by affective arousal / novelty and fatigue proxies:
        d_tau/dt = base_rate * (1 + 0.6 * arousal + 0.4 * novelty - 0.5 * fatigue)
    """

    def __init__(self, *, base_rate: float = 1.0) -> None:
        self.base_rate = float(base_rate)
        self._prev = time.perf_counter()
        self._tau = 0.0
        self._marks: Dict[str, float] = {}

    @property
    def tau(self) -> float:
        return self._tau

    def tau_now(self) -> float:
        return self._tau

    def tick(
        self,
        mood: Optional[Dict[str, float]] = None,
        intero: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        mood = mood or {}
        intero = intero or {}
        now = time.perf_counter()
        dt = max(0.0, now - self._prev)
        self._prev = now

        arousal = float(mood.get("a", mood.get("arousal", 0.5)))
        novelty = float(mood.get("n", mood.get("novelty", 0.2)))
        fatigue = float(intero.get("fatigue", intero.get("fatigue_level", 0.3)))

        rate = self.base_rate * (1.0 + 0.6 * arousal + 0.4 * novelty - 0.5 * fatigue)
        rate = _clip(rate, 0.5, 1.5)

        d_tau = rate * dt
        self._tau += d_tau

        return {
            "mono_ts": now,
            "dt": dt,
            "d_tau": d_tau,
            "tau": self._tau,
            "tau_rate": rate,
        }

    def mark(self, key: str) -> None:
        """Remember current tau for a named event."""
        self._marks[key] = self._tau

    def since_last(self, key: str) -> float:
        """Return elapsed tau since mark(key); large number if never marked."""
        if key not in self._marks:
            return float("inf")
        return self._tau - self._marks[key]

    def mark(self, key: str) -> None:
        self._marks[key] = self._tau

    def since(self, key: str) -> float:
        return self._tau - self._marks.get(key, self._tau)


__all__ = ["TimeKeeper"]
