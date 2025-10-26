# -*- coding: utf-8 -*-
"""Sensitivity heuristics for control parameters."""

from __future__ import annotations

from typing import Dict, Mapping


def simple_sensitivity(contributions: Mapping[str, float], eps: Mapping[str, float]) -> Dict[str, float]:
    """
    Estimate sensitivity as contribution / epsilon for configured axes.
    """
    sensitivities: Dict[str, float] = {}
    for key, epsilon in eps.items():
        try:
            eps_val = float(epsilon)
        except (TypeError, ValueError):
            continue
        if abs(eps_val) < 1e-9:
            continue
        delta = float(contributions.get(key, 0.0))
        sensitivities[key] = delta / eps_val
    return sensitivities


__all__ = ["simple_sensitivity"]
