from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time

import numpy as np


@dataclass
class TemporalWeightState:
    value: float = 0.0
    last_ts: Optional[float] = None


class TemporalWeightCore:
    """Small temporal accumulator for lingering affect, cue pressure, or load."""

    def __init__(
        self,
        *,
        decay_per_s: float = 0.92,
        rise_gain: float = 1.0,
        clamp_low: float = 0.0,
        clamp_high: float = 1.0,
    ) -> None:
        self.decay_per_s = float(decay_per_s)
        self.rise_gain = float(rise_gain)
        self.clamp_low = float(clamp_low)
        self.clamp_high = float(clamp_high)
        self.state = TemporalWeightState(value=0.0, last_ts=time.time())

    def decay_only(self, *, now: Optional[float] = None) -> float:
        now_ts = float(now if now is not None else time.time())
        prev_ts = self.state.last_ts if self.state.last_ts is not None else now_ts
        dt = max(now_ts - prev_ts, 0.0)
        self.state.last_ts = now_ts
        self.state.value = float(
            np.clip(
                self.state.value * (max(self.decay_per_s, 0.0) ** dt),
                self.clamp_low,
                self.clamp_high,
            )
        )
        return self.state.value

    def push(self, amount: float, *, now: Optional[float] = None) -> float:
        base = self.decay_only(now=now)
        self.state.value = float(
            np.clip(
                base + self.rise_gain * float(amount),
                self.clamp_low,
                self.clamp_high,
            )
        )
        return self.state.value

    def reignite(self, cue_strength: float, *, linger_bias: float = 0.35) -> float:
        current = self.state.value
        target = max(current, 0.0) * float(linger_bias) + float(cue_strength)
        self.state.value = float(np.clip(target, self.clamp_low, self.clamp_high))
        return self.state.value

