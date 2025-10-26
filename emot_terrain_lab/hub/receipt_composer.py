# -*- coding: utf-8 -*-
"""Helper for enriching hub receipts."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional


def augment_receipt(
    receipt: Dict[str, object],
    *,
    coherence: Optional[float],
    persona: Dict[str, object],
    safety: Dict[str, object],
    value_weights: Dict[str, float],
    biofield: Optional[Dict[str, object]] = None,
    control_deltas: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, object]:
    """Attach narrative, persona, safety, and value weight details to receipt."""
    if coherence is not None:
        receipt["narrative"] = {"coherence": round(float(coherence), 3)}
    else:
        receipt["narrative"] = {"coherence": None}
    receipt["value_weights"] = dict(value_weights)
    receipt["bayes"] = {
        "persona": persona,
        "safety": safety,
    }
    if biofield is not None:
        receipt["biofield"] = biofield
    if control_deltas is not None:
        receipt.setdefault("controls_delta", control_deltas)
    return receipt


def diff_controls(stages: Mapping[str, Mapping[str, float]]) -> Dict[str, Dict[str, float]]:
    """Compute per-stage control differences."""
    if "base" not in stages:
        return {}
    ordered: Iterable[str] = [key for key in stages.keys()]
    stage_names = [name for name in ordered if name != "base"]
    keys = set(stages["base"].keys())
    for stage in stage_names:
        keys.update(stages[stage].keys())
    diffs: Dict[str, Dict[str, float]] = {stage: {} for stage in stage_names}
    total: Dict[str, float] = {}
    base_controls = stages["base"]
    for key in keys:
        prev = float(base_controls.get(key, 0.0))
        base_value = prev
        for stage in stage_names:
            current_stage = stages.get(stage, {})
            curr = float(current_stage.get(key, prev))
            delta = curr - prev
            if abs(delta) > 1e-9:
                diffs[stage][key] = round(delta, 4)
            prev = curr
        total_delta = prev - base_value
        if abs(total_delta) > 1e-9:
            total[key] = round(total_delta, 4)
    diffs["total"] = total
    return diffs


def verify_contributions(
    stages: Mapping[str, Mapping[str, float]],
    diffs: Mapping[str, Mapping[str, float]],
    tol: float = 1e-3,
) -> bool:
    """Ensure per-stage deltas reconcile with final controls."""
    names = [name for name in stages.keys() if name != "base"]
    if not names:
        return True
    base = stages.get("base", {})
    prev = dict(base)
    for name in names:
        stage = stages.get(name, {})
        diff = diffs.get(name, {})
        keys = set(stage.keys()) | set(prev.keys()) | set(diff.keys())
        for key in keys:
            expected = float(stage.get(key, prev.get(key, 0.0))) - float(prev.get(key, 0.0))
            actual = float(diff.get(key, 0.0))
            if abs(round(expected - actual, 4)) > tol:
                return False
        prev = dict(stage)
    final = prev
    total_diff = diffs.get("total", {})
    keys = set(final.keys()) | set(base.keys()) | set(total_diff.keys())
    for key in keys:
        expected = float(final.get(key, base.get(key, 0.0))) - float(base.get(key, 0.0))
        actual = float(total_diff.get(key, 0.0))
        if abs(round(expected - actual, 4)) > tol:
            return False
    return True


__all__ = ["augment_receipt", "diff_controls", "verify_contributions"]
