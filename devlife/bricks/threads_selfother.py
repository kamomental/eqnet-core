"""Classify events as self-generated or other-generated."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Dict, Tuple
from collections import deque

import numpy as np


@dataclass
class SelfOtherConfig:
    threshold: float = 0.15
    cooldown_steps: int = 5
    deadband: float = 0.05     # neutral zone to avoid flicker
    vote_window: int = 5       # majority vote window


@dataclass
class SelfOtherClassifier:
    config: SelfOtherConfig = field(default_factory=SelfOtherConfig)
    last_self: float = 0.0
    last_other: float = 0.0
    cooldown: int = 0
    _votes: Deque[int] = field(default_factory=lambda: deque(maxlen=5))

    def classify(self, sigma: float, psi: float, external_event: int) -> Dict[str, int]:
        delta_self = abs(sigma - self.last_self) + abs(psi - self.last_other)
        self.last_self = sigma
        self.last_other = psi
        self.cooldown = max(0, self.cooldown - 1)
        vote_other = 0
        if external_event and self.cooldown == 0:
            # deadband: if delta is very small, treat as neutral (0)
            if delta_self <= self.config.deadband:
                vote_other = 0
            else:
                vote_other = 1
            self.cooldown = self.config.cooldown_steps
        # append vote and compute majority
        if self._votes.maxlen != self.config.vote_window:
            self._votes = deque(self._votes, maxlen=self.config.vote_window)
        self._votes.append(vote_other)
        maj_other = 1 if sum(self._votes) > (len(self._votes) // 2) else 0
        return {"self": int(maj_other == 0), "other": maj_other, "conflict": int(external_event and delta_self > self.config.threshold)}
