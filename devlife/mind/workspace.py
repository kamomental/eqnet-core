from __future__ import annotations

import numpy as np


class WMBuffer:
    """Exponential-decay working memory occupancy tracker.

    update(importance, dt_ms) returns the current occupancy s in [0,1].
    """

    def __init__(self, tau_ms: float = 4000.0) -> None:
        self.tau = float(max(1.0, tau_ms))
        self.s = 0.0

    def update(self, importance: float, dt_ms: float = 200.0) -> float:
        decay = float(np.exp(-dt_ms / self.tau))
        self.s = self.s * decay + float(np.clip(importance, 0.0, 1.0)) * (1.0 - decay)
        return float(np.clip(self.s, 0.0, 1.0))


__all__ = ["WMBuffer"]

