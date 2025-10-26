# -*- coding: utf-8 -*-
"""Minimal online learner for persona preferences."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple


def _clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def update_preferences(
    preferences: Mapping[str, float],
    feedback: Mapping[str, float],
    *,
    lr: float = 0.1,
    momentum: float = 0.0,
) -> Dict[str, float]:
    """Apply bandit-style update based on axis feedback."""
    updated: Dict[str, float] = {}
    for axis in set(preferences.keys()) | set(feedback.keys()):
        prev = float(preferences.get(axis, 0.0))
        delta = float(feedback.get(axis, 0.0))
        value = prev * (1.0 - momentum) + lr * delta
        updated[axis] = _clip(prev + value)
    return updated


def feedback_from_labels(
    label: str,
    intensity: float = 1.0,
) -> Dict[str, float]:
    """Map coarse feedback labels to axis deltas."""
    label = label.lower()
    if label in {"more_direct", "direct"}:
        return {"directness": 1.0 * intensity, "hedging": -0.5 * intensity}
    if label in {"softer", "gentle"}:
        return {"directness": -0.7 * intensity, "hedging": 0.6 * intensity}
    if label in {"more_formal"}:
        return {"formality": 0.8 * intensity, "emoji_use": -0.5 * intensity}
    if label in {"more_fun", "playful"}:
        return {"humor": 0.8 * intensity, "emoji_use": 0.4 * intensity}
    if label in {"calm_down"}:
        return {"tempo": -0.8 * intensity, "rhythm_pause": 0.5 * intensity}
    if label in {"speed_up"}:
        return {"tempo": 0.8 * intensity, "rhythm_pause": -0.6 * intensity}
    return {}


__all__ = ["update_preferences", "feedback_from_labels"]
