# -*- coding: utf-8 -*-
"""Catalyst detection between membrane and field."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, List

import numpy as np


@dataclass
class CatalystParams:
    gradient_threshold: float = 0.35
    memory_threshold: float = 0.25
    cooldown_steps: int = 5
    activate_gain: float = 0.8
    soothe_gain: float = -0.6


@dataclass
class CatalystEvent:
    timestamp: str
    locale: str
    position: List[float]
    reactiveness: float
    node: Optional[str]
    mode: str  # "activate" or "soothe"

    def to_json(self) -> Dict:
        return asdict(self)


class CatalystManager:
    """Evaluate and apply catalyst reactions based on membrane-field interaction."""

    def __init__(self, params: CatalystParams | None = None) -> None:
        self.params = params or CatalystParams()
        self.events: List[Dict] = []
        self._cooldown = 0

    def to_json(self) -> List[Dict]:
        return list(self.events)

    @staticmethod
    def from_json(payload: List[Dict]) -> "CatalystManager":
        manager = CatalystManager()
        manager.events = payload
        return manager

    def log_event(self, event: CatalystEvent) -> None:
        self.events.append(event.to_json())

    def evaluate(
        self,
        timestamp: str,
        locale: str,
        emotion_vec: np.ndarray,
        membrane_state: Dict[str, float],
        field_snapshot: Dict[str, np.ndarray],
        field,
        memory_palace,
    ) -> Optional[CatalystEvent]:
        if self._cooldown > 0:
            self._cooldown -= 1
            return None

        grad_mag = self._sample_gradient(field_snapshot["magnitude"], emotion_vec, field)
        reactiveness = grad_mag * (1.0 + membrane_state.get("a", 0.0)) * (1.0 + abs(membrane_state.get("delta_tau", 0.0)))
        if reactiveness < self.params.gradient_threshold:
            return None

        position = np.array(field.coords_for_emotion(emotion_vec))
        node_name, memory_score = memory_palace.strongest_alignment(locale, position)
        if memory_score < self.params.memory_threshold:
            return None

        total_react = reactiveness * (1.0 + memory_score)
        mode = "activate" if membrane_state.get("delta_tau", 0.0) >= 0 else "soothe"
        event = CatalystEvent(
            timestamp=timestamp,
            locale=locale,
            position=position.tolist(),
            reactiveness=float(total_react),
            node=node_name,
            mode=mode,
        )
        self._cooldown = self.params.cooldown_steps
        self.log_event(event)
        self._apply_field_effect(field, event)
        return event

    def _apply_field_effect(self, field, event: CatalystEvent) -> None:
        gain = self.params.activate_gain if event.mode == "activate" else self.params.soothe_gain
        x_norm = event.position[0] / max(field.params.grid_size - 1, 1)
        y_norm = event.position[1] / max(field.params.grid_size - 1, 1)
        field.inject_explicit([(float(x_norm), float(y_norm), gain)])

    def _sample_gradient(self, magnitude: np.ndarray, emotion_vec: np.ndarray, field) -> float:
        x, y = field.coords_for_emotion(emotion_vec)
        return float(field.sample_scalar(magnitude, x, y))
