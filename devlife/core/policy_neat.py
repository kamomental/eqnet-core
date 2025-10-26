"""Simplified policy placeholder (NEAT-inspired linear mapping)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

import numpy as np


@dataclass
class PolicyConfig:
    sensor_dim: int = 64
    hormone_dim: int = 3
    actuator_dim: int = 8
    seed: int = 0


class SimplePolicy:
    """Linear policy mixing sensors and hormones to produce actuator map."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()
        rng = np.random.default_rng(self.config.seed)
        self.W_s = rng.normal(scale=0.25, size=(self.config.actuator_dim, self.config.sensor_dim)).astype(np.float32)
        self.W_h = rng.normal(scale=0.2, size=(self.config.actuator_dim, self.config.hormone_dim)).astype(np.float32)
        self.bias = rng.normal(scale=0.05, size=self.config.actuator_dim).astype(np.float32)

    def act(self, sensors: Dict[str, Any], hormones: Dict[str, float]) -> Dict[str, float]:
        flat_sensors = self._flatten_sensors(sensors)
        vector_h = np.array([hormones.get(k, 0.0) for k in sorted(hormones.keys())], dtype=np.float32)
        flat_sensors = self._pad(flat_sensors, self.W_s.shape[1])
        vector_h = self._pad(vector_h, self.W_h.shape[1])
        logits = self.W_s @ flat_sensors + self.W_h @ vector_h + self.bias
        activation = np.tanh(logits)
        return {f"a{i}": float(value) for i, value in enumerate(activation)}

    # ------------------------------------------------------------------ helpers
    def _flatten_sensors(self, sensors: Dict[str, Any]) -> np.ndarray:
        values: Iterable[np.ndarray] = []
        buffer: list[np.ndarray] = []

        def visit(obj: Any) -> None:
            if isinstance(obj, dict):
                for key in sorted(obj.keys()):
                    visit(obj[key])
            elif isinstance(obj, np.ndarray):
                buffer.append(obj.flatten().astype(np.float32))
            elif np.isscalar(obj):
                buffer.append(np.array([obj], dtype=np.float32))
            elif hasattr(obj, "flatten"):
                buffer.append(np.asarray(obj, dtype=np.float32).flatten())

        visit(sensors)
        if not buffer:
            return np.zeros(self.config.sensor_dim, dtype=np.float32)
        concatenated = np.concatenate(buffer)
        return concatenated.astype(np.float32)

    def _pad(self, vector: np.ndarray, size: int) -> np.ndarray:
        if vector.size < size:
            return np.pad(vector, (0, size - vector.size))
        return vector[:size]
