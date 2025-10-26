# -*- coding: utf-8 -*-
"""Composition metrics for control-stage pipelines."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable


def _apply_delta(base: Dict[str, float], delta: Dict[str, float]) -> Dict[str, float]:
    updated = deepcopy(base)
    for key, value in delta.items():
        updated[key] = float(updated.get(key, 0.0)) + float(value)
    return updated


def assoc_defect(
    base: Dict[str, float],
    persona_stage: Dict[str, float],
    norms_stage: Dict[str, float],
    green_stage: Dict[str, float],
) -> tuple[float, Dict[str, float]]:
    """
    Estimate associativity defect between (persona ∘ norms) ∘ green and persona ∘ (norms ∘ green).
    """
    persona_delta = {k: float(persona_stage.get(k, 0.0)) - float(base.get(k, 0.0)) for k in _keys(base, persona_stage)}
    norms_delta = {
        k: float(norms_stage.get(k, persona_stage.get(k, 0.0))) - float(persona_stage.get(k, 0.0))
        for k in _keys(persona_stage, norms_stage)
    }
    green_delta = {
        k: float(green_stage.get(k, norms_stage.get(k, 0.0))) - float(norms_stage.get(k, 0.0))
        for k in _keys(norms_stage, green_stage)
    }

    left = _apply_delta(base, persona_delta)
    left = _apply_delta(left, norms_delta)
    left = _apply_delta(left, green_delta)

    norms_then_green = _apply_delta(base, norms_delta)
    norms_then_green = _apply_delta(norms_then_green, green_delta)
    right = _apply_delta(norms_then_green, persona_delta)

    diff = {key: left.get(key, 0.0) - right.get(key, 0.0) for key in set(left) | set(right)}
    defect = sum(value * value for value in diff.values()) ** 0.5
    return defect, diff


def _keys(*dicts: Dict[str, float]) -> Iterable[str]:
    keys = set()
    for mapping in dicts:
        keys.update(mapping.keys())
    return keys


__all__ = ["assoc_defect"]
