# -*- coding: utf-8 -*-
"""Continuous emotional energy field (EQNet) prototype.

The field keeps three bounded thermodynamic-style indicators that downstream
controllers can rely on without re-deriving the maths each time:

* ``S`` — entropy normalised to ``[0, 1]`` (higher = more disorder)
* ``H`` — enthalpy/energy carrier normalised to ``[0, 1]``
* ``ρ`` — synchrony/density proxy derived from gradient magnitudes (higher =
  calmer flow, lower gradients)

Together with a reward delta ``ΔR`` (see :mod:`devlife.runtime.loop`), the
Ignition Index is anchored to the contract::

    I = w₁·ΔR_norm + w₂·(1 − S)
    w₁ + w₂ = 1,   S ∈ [0, 1]

which guarantees ``∂I/∂S ≤ 0`` and ``∂I/∂ΔR ≥ 0`` when the weights are
non-negative.  This file focuses on producing stable S/H/ρ signals so that the
loop only has to reason about combining them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .config import load_culture_projection, project_emotion
from .emotion import AXES


def detect_buds_2d(
    field: np.ndarray,
    min_prominence: float = 0.08,
    max_points: int = 32,
) -> List[Tuple[int, int]]:
    """
    Detect small peaks that may correspond to budding qualia clusters.

    Returns a list of (x, y) coordinates sorted by descending field strength.
    """
    surface = np.asarray(field, dtype=float)
    if surface.ndim != 2:
        raise ValueError("field must be 2D")
    mean = float(surface.mean())
    std = float(surface.std())
    if std == 0.0:
        return []
    threshold = mean + min_prominence * std
    mask = surface > threshold
    if not mask.any():
        return []

    padded = np.pad(surface, 1, mode="edge")
    centre = padded[1:-1, 1:-1]
    neighbour_max = np.maximum.reduce(
        [
            padded[:-2, 1:-1],
            padded[2:, 1:-1],
            padded[1:-1, :-2],
            padded[1:-1, 2:],
            padded[:-2, :-2],
            padded[:-2, 2:],
            padded[2:, :-2],
            padded[2:, 2:],
        ]
    )
    local_peaks = (centre >= neighbour_max) & mask
    coords = np.argwhere(local_peaks)
    if coords.size == 0:
        return []
    strengths = surface[coords[:, 0], coords[:, 1]]
    order = np.argsort(strengths)
    if len(order) > max_points:
        order = order[-max_points:]
    selected = coords[order]
    return [(int(x), int(y)) for x, y in selected.tolist()]


@dataclass
class FieldParams:
    grid_size: int = 64
    alpha: float = 0.35   # diffusion
    lam: float = 0.08     # dissipation
    dt: float = 0.5
    memory_decay: float = 0.985
    memory_gain: float = 0.03
    source_sigma: float = 2.5
    locale: str = "default"
    enthalpy_gain: float = 0.6


class EmotionField:
    """Emotion energy field governed by a diffusion-like PDE."""

    def __init__(
        self,
        params: FieldParams | None = None,
        energy: np.ndarray | None = None,
        memory: np.ndarray | None = None,
    ) -> None:
        self.params = params or FieldParams()
        shape = (self.params.grid_size, self.params.grid_size)
        self.energy = np.zeros(shape, dtype=float) if energy is None else energy
        self.memory = np.zeros(shape, dtype=float) if memory is None else memory
        self._pending_sources: List[np.ndarray] = []
        self._grid_coords = np.indices(shape)
        self._config_path = Path("config/culture.yaml")
        self._projection_matrix = load_culture_projection(self._config_path).matrix(self.params.locale)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def to_json(self) -> Dict:
        return {
            "params": asdict(self.params),
            "energy": self.energy.tolist(),
            "memory": self.memory.tolist(),
        }

    @staticmethod
    def from_json(payload: Dict) -> "EmotionField":
        params = FieldParams(**payload.get("params", {}))
        energy = np.array(payload.get("energy", []), dtype=float)
        memory = np.array(payload.get("memory", []), dtype=float)
        if energy.size == 0:
            energy = None
        if memory.size == 0:
            memory = None
        field = EmotionField(params=params, energy=energy, memory=memory)
        field.set_locale(params.locale)
        return field

    # ------------------------------------------------------------------ #
    # Source management
    # ------------------------------------------------------------------ #

    def inject_emotion(self, emotion_vec: np.ndarray, weight: float = 1.0) -> None:
        """Convert 9D emotion vector into a localized source."""
        x, y = self._emotion_to_grid_coords(emotion_vec)
        intensity = float(np.linalg.norm(emotion_vec)) * weight
        patch = self._gaussian_patch(x, y, intensity)
        self._pending_sources.append(patch)

    def inject_explicit(self, points: Iterable[Tuple[float, float, float]]) -> None:
        """Inject explicit sources (x,y,intensity) with coordinates in [0,1]."""
        for x, y, v in points:
            gx = x * (self.params.grid_size - 1)
            gy = y * (self.params.grid_size - 1)
            self._pending_sources.append(self._gaussian_patch(gx, gy, v))

    # ------------------------------------------------------------------ #
    # Evolution
    # ------------------------------------------------------------------ #

    def step(self, steps: int = 1) -> None:
        for _ in range(steps):
            source = self._aggregate_sources()
            lap = self._laplacian(self.energy)
            self.energy += self.params.dt * (
                self.params.alpha * lap
                - self.params.lam * self.energy
                + source
                + self.memory
            )
            self.memory *= self.params.memory_decay
            self.memory += self.params.memory_gain * np.clip(source, 0.0, None)

    # ------------------------------------------------------------------ #
    # Diagnostics / snapshots
    # ------------------------------------------------------------------ #

    def snapshot(self) -> Dict[str, np.ndarray]:
        grad_x, grad_y = np.gradient(self.energy)
        flow_x = -grad_x
        flow_y = -grad_y
        magnitude = np.sqrt(flow_x ** 2 + flow_y ** 2)
        phase = np.arctan2(flow_y, flow_x)
        return {
            "energy": self.energy.copy(),
            "flow_x": flow_x,
            "flow_y": flow_y,
            "magnitude": magnitude,
            "phase": phase,
        }

    def set_locale(self, locale: str) -> None:
        self.params.locale = locale
        self._projection_matrix = load_culture_projection(self._config_path).matrix(locale)

    def feedback_vector(self, emotion_vec: np.ndarray) -> np.ndarray:
        """Compute a 9D feedback vector derived from local field gradients."""
        x, y = self.coords_for_emotion(emotion_vec)
        grad_x, grad_y = self._sample_gradient(x, y)
        grad2d = np.array([grad_x, grad_y], dtype=float)
        if not np.any(np.isfinite(grad2d)):
            return np.zeros(len(AXES), dtype=float)
        pinv = np.linalg.pinv(self._projection_matrix)
        feedback = pinv @ grad2d
        return np.nan_to_num(feedback, nan=0.0)

    def coords_for_emotion(self, emotion_vec: np.ndarray) -> Tuple[float, float]:
        return self._emotion_to_grid_coords(emotion_vec)

    def sample_scalar(self, arr: np.ndarray, x: float, y: float) -> float:
        return self._bilinear_sample(arr, x, y)

    def compute_metrics(self) -> Dict[str, float]:
        """Return bounded thermodynamic summaries of the current field."""
        snapshot = self.snapshot()
        energy = np.nan_to_num(snapshot["energy"], nan=0.0, posinf=0.0, neginf=0.0)
        shift = energy - np.nanmax(energy)
        exp = np.exp(shift)
        exp_sum = exp.sum() + 1e-12
        prob = exp / exp_sum
        raw_entropy = float(-np.sum(prob * np.log(prob + 1e-12)))
        grid_size = max(2, energy.size)
        entropy_cap = math.log(grid_size)
        entropy_norm = float(np.clip(raw_entropy / entropy_cap, 0.0, 1.0))

        dissipation = float(self.params.lam * np.sum(np.abs(energy)))
        info_flux = float(np.mean(np.nan_to_num(snapshot["magnitude"], nan=0.0)))

        enthalpy = np.nan_to_num(energy + self.params.enthalpy_gain * self.memory, nan=0.0)
        enthalpy_mean = float(enthalpy.mean())
        enthalpy_var = float(enthalpy.var())
        enthalpy_norm = float(np.clip(0.5 + 0.5 * np.tanh(enthalpy_mean), 0.0, 1.0))

        grad_x, grad_y = np.gradient(energy)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_mean = float(np.mean(np.nan_to_num(grad_mag, nan=0.0)))
        rho_norm = float(np.clip(1.0 - math.tanh(grad_mean), 0.0, 1.0))

        energy_mean = float(energy.mean())
        energy_var = float(energy.var())
        return {
            "entropy": raw_entropy,
            "entropy_norm": entropy_norm,
            "enthalpy_norm": enthalpy_norm,
            "rho_norm": rho_norm,
            "dissipation": dissipation,
            "info_flux": info_flux,
            "energy_mean": energy_mean,
            "energy_var": energy_var,
            "enthalpy_mean": enthalpy_mean,
            "enthalpy_var": enthalpy_var,
            "grad_mean": grad_mean,
        }

    def qualia_signature(
        self,
        emotion_vec: np.ndarray,
        snapshot: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """
        Sample local field properties (qualia) for a given emotion vector.

        Parameters
        ----------
        emotion_vec:
            9D emotion state to project onto the field grid.
        snapshot:
            Optional pre-computed snapshot dictionary produced by `snapshot()`.
            Supplying this avoids recomputing gradients when multiple samples
            are required within the same tick.
        """
        snap = snapshot or self.snapshot()
        x, y = self.coords_for_emotion(emotion_vec)
        energy = np.nan_to_num(self.sample_scalar(snap["energy"], x, y), nan=0.0)
        magnitude = np.nan_to_num(self.sample_scalar(snap["magnitude"], x, y), nan=0.0)
        phase = np.nan_to_num(self.sample_scalar(snap["phase"], x, y), nan=0.0)
        flow_x = np.nan_to_num(self.sample_scalar(snap["flow_x"], x, y), nan=0.0)
        flow_y = np.nan_to_num(self.sample_scalar(snap["flow_y"], x, y), nan=0.0)
        memory = np.nan_to_num(self.sample_scalar(self.memory, x, y), nan=0.0)
        enthalpy = float(energy + self.params.enthalpy_gain * memory)
        return {
            "energy": float(energy),
            "magnitude": float(magnitude),
            "phase": float(phase),
            "flow": [float(flow_x), float(flow_y)],
            "memory": float(memory),
            "enthalpy": enthalpy,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _aggregate_sources(self) -> np.ndarray:
        if not self._pending_sources:
            return np.zeros_like(self.energy)
        source = np.sum(self._pending_sources, axis=0)
        self._pending_sources.clear()
        return source

    def _laplacian(self, field: np.ndarray) -> np.ndarray:
        padded = np.pad(field, 1, mode="edge")
        centre = field
        lap = (
            padded[1:-1, :-2]
            + padded[1:-1, 2:]
            + padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            - 4 * centre
        )
        return lap

    def _gaussian_patch(self, x: float, y: float, strength: float) -> np.ndarray:
        sigma = self.params.source_sigma
        size = int(max(3, math.ceil(3 * sigma)))
        ix = int(round(x))
        iy = int(round(y))
        x_min = max(0, ix - size)
        x_max = min(self.params.grid_size - 1, ix + size)
        y_min = max(0, iy - size)
        y_max = min(self.params.grid_size - 1, iy + size)

        patch = np.zeros_like(self.energy)
        if x_min > x_max or y_min > y_max:
            return patch

        xs = np.arange(x_min, x_max + 1)
        ys = np.arange(y_min, y_max + 1)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        patch_region = np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * sigma ** 2))
        patch[x_min : x_max + 1, y_min : y_max + 1] = strength * patch_region / (2 * math.pi * sigma ** 2)
        return patch

    def _emotion_to_grid_coords(self, emotion_vec: np.ndarray) -> Tuple[float, float]:
        coords = project_emotion(emotion_vec, self.params.locale)
        norm = 1 / (1 + np.exp(-np.clip(coords, -6, 6)))
        x = float(norm[0]) * (self.params.grid_size - 1)
        y = float(norm[1]) * (self.params.grid_size - 1)
        return x, y

    def _sample_gradient(self, x: float, y: float) -> Tuple[float, float]:
        grad_x, grad_y = np.gradient(self.energy)
        gx = self._bilinear_sample(grad_x, x, y)
        gy = self._bilinear_sample(grad_y, x, y)
        return gx, gy

    def _bilinear_sample(self, arr: np.ndarray, x: float, y: float) -> float:
        max_index = self.params.grid_size - 1
        x = float(np.clip(x, 0.0, max_index))
        y = float(np.clip(y, 0.0, max_index))
        x0 = int(math.floor(x))
        x1 = min(x0 + 1, max_index)
        y0 = int(math.floor(y))
        y1 = min(y0 + 1, max_index)
        dx = x - x0
        dy = y - y0
        v00 = arr[x0, y0]
        v10 = arr[x1, y0]
        v01 = arr[x0, y1]
        v11 = arr[x1, y1]
        return (
            v00 * (1 - dx) * (1 - dy)
            + v10 * dx * (1 - dy)
            + v01 * (1 - dx) * dy
            + v11 * dx * dy
        )
