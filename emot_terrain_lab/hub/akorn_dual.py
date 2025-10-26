from __future__ import annotations

import math


class DualOsc:
    """Minimal dual-oscillator for session/affect rhythms.

    Produces a composite R in [0,1] by mixing a base oscillator and an affect-driven
    oscillator. Intended as a light-weight input to higher-level gating.
    """

    def __init__(self, w_base: float = 0.4, w_aff: float = 0.6) -> None:
        self.wb = float(max(0.0, min(1.0, w_base)))
        self.wa = float(max(0.0, min(1.0, w_aff)))
        self.theta_b = 0.0
        self.theta_a = 0.0

    def step(self, omega_base: float, omega_aff: float) -> tuple[float, float, float]:
        self.theta_b += float(omega_base)
        self.theta_a += float(omega_aff)
        Rb = math.cos(self.theta_b)
        Ra = math.cos(self.theta_a)
        mixed = self.wb * Rb + self.wa * Ra
        R = float(max(0.0, min(1.0, mixed)))
        return R, Rb, Ra


__all__ = ["DualOsc"]
