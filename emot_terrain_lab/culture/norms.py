# -*- coding: utf-8 -*-
"""Basic deontic gate for politeness and humility norms."""

from __future__ import annotations

from typing import Any, Dict


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def deontic_gate(plan: Dict[str, Any], norms: Dict[str, float]) -> Dict[str, Any]:
    """Inject lightweight normative adjustments into the plan."""
    adjusted = dict(plan)
    controls = dict(plan.get("controls", {}))
    politeness = float(norms.get("politeness", 0.0))
    humility = float(norms.get("humility", 0.0))

    if politeness > 0.0:
        controls["directness"] = _clip(
            controls.get("directness", 0.0) - 0.05 * politeness,
            -0.4,
            0.4,
        )
    if humility > 0.0:
        pause = controls.get("pause_ms", 0)
        controls["pause_ms"] = int(_clip(pause + 40 * humility, 0, 800))

    adjusted["controls"] = controls
    adjusted["norms"] = {"politeness": politeness, "humility": humility}
    return adjusted



def norms_penalty(action: str, norms: Dict[str, float]) -> float:
    """Return a small penalty when actions contradict politeness/humility norms."""
    politeness = float(norms.get("politeness", 0.0))
    humility = float(norms.get("humility", 0.0))
    penalty = 0.0
    if politeness > 0.7 and action == "clarify":
        penalty += 0.05 * politeness
    if humility > 0.7 and action == "reframe":
        penalty += 0.05 * humility
    return float(min(1.0, max(0.0, penalty)))


__all__ = ["deontic_gate", "norms_penalty"]
