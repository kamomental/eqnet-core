"""Minimal differentiable body model inspired by Lenia/NCA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    radius = size // 2
    grid = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(grid, grid)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


LAPLACIAN_KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


@dataclass
class BodyConfig:
    grid_size: Tuple[int, int] = (64, 64)
    channels: int = 6
    morph_kernel_sigma: float = 1.6
    repair_kernel_sigma: float = 3.2
    decay: float = 0.01


class BodyNCA:
    """Simple cellular automata body with smoothing kernels."""

    def __init__(self, config: BodyConfig | None = None) -> None:
        self.config = config or BodyConfig()
        h, w = self.config.grid_size
        c = self.config.channels
        self.state = np.zeros((c, h, w), dtype=np.float32)
        self._morph_kernel = _gaussian_kernel(5, self.config.morph_kernel_sigma)
        self._repair_kernel = _gaussian_kernel(7, self.config.repair_kernel_sigma)
        self._last_observation: Dict[str, np.ndarray] | None = None

    # ------------------------------------------------------------------ interface
    def step(self, actuators: np.ndarray) -> Dict[str, np.ndarray]:
        """Update cellular state using Lenia-like smoothing and return observation."""
        if actuators is None:
            actuators = np.zeros_like(self.state)
        if actuators.shape != self.state.shape:
            actuators = self._broadcast_actuators(actuators)
        self.state += 0.1 * actuators
        self.state = np.clip(self.state, -1.0, 1.0)
        self._diffuse(self._morph_kernel, gain=0.7)
        self._diffuse(self._repair_kernel, gain=0.3)
        self.state -= self.config.decay * self.state
        observation = self._make_observation()
        self._last_observation = observation
        return observation

    def observe(self) -> Dict[str, np.ndarray]:
        if self._last_observation is None:
            self._last_observation = self._make_observation()
        return self._last_observation

    def state_stats(self) -> Dict[str, float]:
        return {
            "mean": float(self.state.mean()),
            "std": float(self.state.std()),
            "energy": float(np.square(self.state).mean()),
        }

    def snapshot(self) -> np.ndarray:
        return self.state.copy()

    def damage(self, mask: np.ndarray, severity: float = 1.0) -> None:
        mask = np.broadcast_to(mask, self.state.shape)
        self.state = np.where(mask > 0.5, self.state * (1.0 - severity), self.state)

    # ------------------------------------------------------------------ helpers
    def _diffuse(self, kernel: np.ndarray, gain: float) -> None:
        c, h, w = self.state.shape
        pad_y = kernel.shape[0] // 2
        pad_x = kernel.shape[1] // 2
        padded = np.pad(self.state, ((0, 0), (pad_y, pad_y), (pad_x, pad_x)), mode="wrap")
        out = np.zeros_like(self.state)
        for y in range(h):
            for x in range(w):
                patch = padded[:, y : y + kernel.shape[0], x : x + kernel.shape[1]]
                out[:, y, x] = (patch * kernel).sum(axis=(1, 2))
        self.state = (1 - gain) * self.state + gain * out

    def _make_observation(self) -> Dict[str, np.ndarray]:
        stats = self._compute_stats()
        kappa = self._estimate_curvature()
        observations = {
            "stats": stats,
            "kappa": kappa,
            "channels": self.state.copy(),
        }
        return observations

    def _broadcast_actuators(self, actuators: np.ndarray) -> np.ndarray:
        if actuators.ndim == 1:
            actuators = actuators[:, None, None]
        if actuators.shape[1:] == (1, 1):
            return np.broadcast_to(actuators, self.state.shape)
        raise ValueError(f"Cannot broadcast actuators shape {actuators.shape} to body state {self.state.shape}")

    def _compute_stats(self) -> Dict[str, float]:
        channel0 = self.state[0]
        mean = float(channel0.mean())
        var = float(channel0.var())
        # approximate edge magnitude using finite differences
        dy = np.roll(channel0, -1, axis=0) - channel0
        dx = np.roll(channel0, -1, axis=1) - channel0
        edge = float(np.mean(np.abs(dx)) + np.mean(np.abs(dy)))
        return {"mean": mean, "var": var, "edge": edge}

    def _estimate_curvature(self) -> float:
        channel0 = self.state[0]
        padded = np.pad(channel0, 1, mode="wrap")
        lap = np.zeros_like(channel0)
        for y in range(channel0.shape[0]):
            for x in range(channel0.shape[1]):
                patch = padded[y : y + 3, x : x + 3]
                lap[y, x] = np.sum(patch * LAPLACIAN_KERNEL)
        return float(np.mean(np.abs(lap)))
