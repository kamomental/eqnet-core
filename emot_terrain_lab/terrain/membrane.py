# -*- coding: utf-8 -*-
"""Membrane controller that adapts (r, a, Δτ) via empowerment heuristics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np

from .field import EmotionField


@dataclass
class MembraneState:
    r: float = 1.5
    a: float = 0.5
    delta_tau: float = 0.0
    empowerment: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


class MembraneController:
    """Adaptive controller for the qualia membrane."""

    def __init__(self) -> None:
        self.state = MembraneState()
        self._smoothing = 0.7

    def to_json(self) -> Dict[str, float]:
        return self.state.as_dict()

    @staticmethod
    def from_json(payload: Dict[str, float]) -> "MembraneController":
        ctrl = MembraneController()
        ctrl.state = MembraneState(
            r=float(payload.get("r", 1.5)),
            a=float(payload.get("a", 0.5)),
            delta_tau=float(payload.get("delta_tau", 0.0)),
            empowerment=float(payload.get("empowerment", 0.0)),
        )
        return ctrl

    def update(self, emotion_vec: np.ndarray, field: EmotionField) -> MembraneState:
        snapshot = field.snapshot()
        x, y = field.coords_for_emotion(emotion_vec)
        empowerment_raw = field.sample_scalar(snapshot["magnitude"], x, y)
        energy_local = field.sample_scalar(snapshot["energy"], x, y)
        energy_mean = float(snapshot["energy"].mean())
        empowerment = np.clip(empowerment_raw, 0.0, None)

        # Adaptive recurrence: higher when empowerment is low or local energy deviates strongly.
        delta_energy = np.tanh((energy_local - energy_mean))
        new_delta_tau = self._smoothing * self.state.delta_tau + (1 - self._smoothing) * float(delta_energy)

        target_a = float(np.tanh(empowerment))
        new_a = self._smoothing * self.state.a + (1 - self._smoothing) * target_a

        target_r = 1.0 + 4.0 * (1.0 - new_a) + 0.8 * abs(new_delta_tau)
        new_r = self._smoothing * self.state.r + (1 - self._smoothing) * target_r

        new_emp = self._smoothing * self.state.empowerment + (1 - self._smoothing) * float(empowerment)

        self.state = MembraneState(
            r=float(np.clip(new_r, 1.0, 8.0)),
            a=float(np.clip(new_a, 0.0, 1.0)),
            delta_tau=float(np.clip(new_delta_tau, -1.0, 1.0)),
            empowerment=float(new_emp),
        )
        return self.state

    def state_dict(self) -> Dict[str, float]:
        return self.state.as_dict()
