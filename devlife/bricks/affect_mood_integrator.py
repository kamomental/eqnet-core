"""Mood integrator with EMA smoothing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import math


def _ema(prev: float, value: float, tau: float, dt: float = 1.0) -> float:
    alpha = dt / max(1e-6, tau)
    alpha = min(1.0, max(0.0, alpha))
    return (1 - alpha) * prev + alpha * value


@dataclass
class MoodConfig:
    tau_valence: float = 90.0
    tau_arousal: float = 60.0
    circadian_amp: float = 0.05
    baseline_v: float = 0.0
    baseline_a: float = 0.0


@dataclass
class MoodIntegrator:
    config: MoodConfig = field(default_factory=MoodConfig)
    mood_v: float = 0.0
    mood_a: float = 0.0
    steps: int = 0

    def update(self, h_valence: float, h_arousal: float, reward: float = 0.0) -> Dict[str, float]:
        circadian = self.config.circadian_amp * math.sin(2 * math.pi * (self.steps % (24 * 60)) / (24 * 60))
        target_v = h_valence + reward + circadian + self.config.baseline_v
        target_a = h_arousal + 0.5 * reward + self.config.baseline_a
        self.mood_v = _ema(self.mood_v, target_v, self.config.tau_valence)
        self.mood_a = _ema(self.mood_a, target_a, self.config.tau_arousal)
        self.steps += 1
        return {"mood_v": self.mood_v, "mood_a": self.mood_a}
