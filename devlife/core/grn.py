"""Lightweight gene regulatory network simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

import numpy as np


@dataclass
class SimpleGRNConfig:
    outputs: Sequence[str] = ("H_valence", "H_arousal", "H_novelty")
    sensory_weights: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "H_valence": {"stats_mean": 0.8, "homeo_error": -0.5},
            "H_arousal": {"stats_edge": 0.6, "novelty_signal": 0.4},
            "H_novelty": {"novelty_signal": 1.0},
        }
    )
    internal_weights: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "H_valence": {"energy": -0.3},
            "H_arousal": {"energy": 0.25},
            "H_novelty": {"kappa": 0.6},
        }
    )
    bias: Dict[str, float] = field(
        default_factory=lambda: {"H_valence": 0.1, "H_arousal": 0.2, "H_novelty": 0.05}
    )
    params: Dict[str, float] = field(default_factory=lambda: {"decay": 0.05, "noise_std": 0.01})
    clamp: Dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "H_valence": (-1.0, 1.0),
            "H_arousal": (0.0, 1.5),
            "H_novelty": (0.0, 2.0),
        }
    )


class SimpleGRN:
    """Gene regulatory network producing homoeostatic hormones."""

    def __init__(self, config: SimpleGRNConfig | None = None) -> None:
        self.config = config or SimpleGRNConfig()
        self.state = {name: 0.0 for name in self.config.outputs}

    def forward(self, internal_state: Mapping[str, float], sensory_stats: Mapping[str, float]) -> Dict[str, float]:
        outputs: Dict[str, float] = {}
        inputs = {**sensory_stats, **internal_state}
        decay = self.config.params.get("decay", 0.0)
        noise_std = self.config.params.get("noise_std", 0.0)

        for hormone in self.config.outputs:
            net = self.config.bias.get(hormone, 0.0)
            for src, weight in self.config.sensory_weights.get(hormone, {}).items():
                net += weight * float(inputs.get(src, 0.0))
            for src, weight in self.config.internal_weights.get(hormone, {}).items():
                net += weight * float(inputs.get(src, 0.0))
            previous = self.state.get(hormone, 0.0)
            z = net - decay * previous
            if noise_std > 0.0:
                z += np.random.normal(scale=noise_std)
            expression = np.tanh(z)
            lo, hi = self.config.clamp.get(hormone, (-np.inf, np.inf))
            clipped = float(np.clip(expression, lo, hi))
            outputs[hormone] = clipped
        self.state.update(outputs)
        return outputs
