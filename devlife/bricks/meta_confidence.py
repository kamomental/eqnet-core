"""Meta-confidence estimation for decisions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class MetaConfig:
    temperature: float = 1.0
    threshold: float = 0.35


@dataclass
class MetaConfidence:
    config: MetaConfig = field(default_factory=MetaConfig)

    def compute(self, logits: np.ndarray, prediction_error: float) -> Dict[str, float | int]:
        scaled = logits / max(1e-6, self.config.temperature)
        probs = np.exp(scaled - scaled.max())
        probs /= probs.sum()
        conf = float(np.max(probs) * (1.0 - prediction_error))
        return {
            "conf": conf,
            "reconsider": int(conf < self.config.threshold),
        }

