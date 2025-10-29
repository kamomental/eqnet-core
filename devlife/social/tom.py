"""Lightweight theory-of-mind with smoothing and rate limiting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class TomSmoothingConfig:
    """Smoothing/robustness knobs for intent trust."""

    alpha: float = 0.2       # EMA coefficient (0.1-0.3 recommended)
    med_window: int = 5      # Median filter window (odd)
    rate_limit: float = 0.15 # Max fractional change per step (e.g., 0.15 = 15%)


@dataclass
class TheoryOfMind:
    """Maintains rough beliefs about a peer's intent."""

    beliefs: Dict[str, float] = field(default_factory=lambda: {"intent_trust": 0.5})
    _hist: List[float] = field(default_factory=list)
    _trust_smoothed: Optional[float] = None
    _cfg: TomSmoothingConfig = field(default_factory=TomSmoothingConfig)

    def update(self, episode: Dict[str, Any]) -> Dict[str, float]:
        p_self = float(episode.get("self_event", 0.0))
        p_other = float(episode.get("other_event", 0.0))
        # Simple heuristic: more "other" activity increases trust
        raw = 0.5 + 0.5 * (p_other - p_self)
        trust = self.update_trust(raw)
        self.beliefs["intent_trust"] = float(np.clip(raw, 0.0, 1.0))
        self.beliefs["intent_trust_smoothed"] = trust
        self.beliefs["last_stage"] = float(p_other)
        return dict(self.beliefs)

    def infer(self) -> Dict[str, float]:
        return dict(self.beliefs)

    # ---------------------------------------------------------- smoothing utils
    def update_trust(self, raw_trust: float) -> float:
        # 1) clip to [0,1]
        x = float(np.clip(raw_trust, 0.0, 1.0))
        # 2) median filter
        self._hist.append(x)
        if len(self._hist) > max(1, int(self._cfg.med_window)):
            self._hist.pop(0)
        med = float(np.median(self._hist))
        # 3) EMA
        if self._trust_smoothed is None:
            ema = med
        else:
            a = float(np.clip(self._cfg.alpha, 0.0, 1.0))
            ema = a * med + (1 - a) * self._trust_smoothed
        # 4) rate limit (fractional change per step)
        if self._trust_smoothed is None:
            limited = ema
        else:
            base = max(1e-6, self._trust_smoothed)
            max_delta = float(self._cfg.rate_limit) * base
            limited = float(np.clip(ema, self._trust_smoothed - max_delta, self._trust_smoothed + max_delta))
        self._trust_smoothed = float(np.clip(limited, 0.0, 1.0))
        return self._trust_smoothed
