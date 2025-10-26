# -*- coding: utf-8 -*-
"""Consent and ethics utilities for emotion terrain system."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .emotion import AXES


@dataclass
class ConsentPreferences:
    record_axes: Dict[str, bool]
    store_dialogue: bool = True
    store_membrane: bool = True
    store_projection: bool = True
    store_field: bool = True
    store_diary: bool = True
    retention_days: int = 30
    allow_forgetting: bool = True
    allow_story_graph: bool = True

    def to_json(self) -> Dict:
        return {
            "record_axes": self.record_axes,
            "store_dialogue": self.store_dialogue,
            "store_membrane": self.store_membrane,
            "store_projection": self.store_projection,
            "store_field": self.store_field,
            "store_diary": self.store_diary,
            "retention_days": self.retention_days,
            "allow_forgetting": self.allow_forgetting,
            "allow_story_graph": self.allow_story_graph,
        }

    @staticmethod
    def default() -> "ConsentPreferences":
        return ConsentPreferences(record_axes={axis: True for axis in AXES})

    @staticmethod
    def from_json(payload: Dict) -> "ConsentPreferences":
        record_axes = payload.get("record_axes", {})
        for axis in AXES:
            record_axes.setdefault(axis, True)
        return ConsentPreferences(
            record_axes=record_axes,
            store_dialogue=payload.get("store_dialogue", True),
            store_membrane=payload.get("store_membrane", True),
            store_projection=payload.get("store_projection", True),
            store_field=payload.get("store_field", True),
            store_diary=payload.get("store_diary", True),
            retention_days=int(payload.get("retention_days", 30)),
            allow_forgetting=payload.get("allow_forgetting", True),
            allow_story_graph=payload.get("allow_story_graph", True),
        )


@dataclass
class BiasReport:
    axis_bias: Dict[str, float]
    record_bias: Dict[str, float]
    notes: str = ""

    def to_json(self) -> Dict:
        return {
            "axis_bias": self.axis_bias,
            "record_bias": self.record_bias,
            "notes": self.notes,
        }


class EthicsManager:
    """Consent management and bias auditing."""

    def __init__(self, preferences: Optional[ConsentPreferences] = None) -> None:
        self.preferences = preferences or ConsentPreferences.default()

    def to_json(self) -> Dict:
        return self.preferences.to_json()

    @staticmethod
    def from_json(payload: Dict) -> "EthicsManager":
        return EthicsManager(ConsentPreferences.from_json(payload))

    def filter_emotion(self, emotion_vec: np.ndarray) -> np.ndarray:
        mask = np.array([self.preferences.record_axes.get(axis, True) for axis in AXES], dtype=bool)
        filtered = np.where(mask, emotion_vec, 0.0)
        return filtered

    def allow_story_graph(self) -> bool:
        return self.preferences.allow_story_graph

    def audit_bias(self, records: Dict[str, int]) -> BiasReport:
        total = sum(records.values()) + 1e-6
        axis_bias = {axis: float(records.get(axis, 0)) / total for axis in AXES}
        record_bias = {axis: float(self.preferences.record_axes.get(axis, True)) for axis in AXES}
        notes = ""
        extremes = [axis for axis, val in axis_bias.items() if val > 0.5]
        if extremes:
            notes = f"High concentration on {extremes}."
        return BiasReport(axis_bias=axis_bias, record_bias=record_bias, notes=notes)

    def update_preferences(self, updates: Dict) -> None:
        self.preferences = ConsentPreferences.from_json(updates)
