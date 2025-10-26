# -*- coding: utf-8 -*-
"""Reverse/forward replay hysteresis controller."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class HysteresisCfg:
    ema_tau: float = 6.0
    hysteresis_up: float = 0.6
    hysteresis_down: float = 0.4
    min_window_tau: float = 0.0
    consecutive_up: int = 1
    consecutive_down: int = 1


class PhaseRatioFilter:
    """Time-aware EMA with hysteresis for replay direction."""

    def __init__(self, cfg: HysteresisCfg) -> None:
        self.cfg = cfg
        self._value: Optional[float] = None
        self._last_tau: Optional[float] = None
        self._direction: str = "mixed"
        self._up_count = 0
        self._down_count = 0

    @property
    def direction(self) -> str:
        return self._direction

    @property
    def value(self) -> Optional[float]:
        return self._value

    def update(self, raw_ratio: float, tau_now: float) -> float:
        """Update filter, returning the stabilized ratio."""
        raw_ratio = max(0.0, min(1.0, float(raw_ratio)))
        tau_now = float(tau_now)

        if self._last_tau is not None:
            if (tau_now - self._last_tau) < max(self.cfg.min_window_tau, 0.0):
                return self._value if self._value is not None else raw_ratio

        if self._value is None or self._last_tau is None:
            filtered = raw_ratio
        else:
            dt = tau_now - self._last_tau
            if dt <= 0.0:
                filtered = self._value
            else:
                alpha = 1.0 - math.exp(-dt / max(self.cfg.ema_tau, 1e-6))
                filtered = alpha * raw_ratio + (1.0 - alpha) * self._value

        self._value = filtered
        self._last_tau = tau_now

        if filtered >= self.cfg.hysteresis_up:
            self._up_count += 1
            self._down_count = 0
            if self._up_count >= max(1, self.cfg.consecutive_up):
                self._direction = "reverse"
        elif filtered <= self.cfg.hysteresis_down:
            self._down_count += 1
            self._up_count = 0
            if self._down_count >= max(1, self.cfg.consecutive_down):
                self._direction = "forward"
        else:
            self._up_count = 0
            self._down_count = 0
            self._direction = "mixed"

        return self._value


__all__ = ["PhaseRatioFilter", "HysteresisCfg"]
