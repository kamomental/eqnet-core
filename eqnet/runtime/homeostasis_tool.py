from __future__ import annotations

from typing import Any, Dict, Mapping


def update_homeostasis(
    *,
    prev_state: Mapping[str, Any] | None,
    resonance: Mapping[str, Any],
    metabolism: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    prev = prev_state if isinstance(prev_state, Mapping) else {}
    prev_arousal = _safe_float(prev.get("arousal_level"), 0.5)
    prev_stability = _safe_float(prev.get("stability_index"), 0.5)
    r_arousal = _safe_float(resonance.get("arousal"), 0.5)
    r_safety = _safe_float(resonance.get("safety"), 0.5)
    r_valence = _safe_float(resonance.get("valence"), 0.5)
    metabolism_map = metabolism if isinstance(metabolism, Mapping) else {}
    attention_level = _safe_float(
        ((metabolism_map.get("resource_budget") or {}).get("attention") or {}).get("level"),
        1.0,
    )
    load = _safe_float(metabolism_map.get("attention_budget_used"), 0.0)
    recovery = _safe_float(
        ((metabolism_map.get("resource_budget") or {}).get("attention") or {}).get("recovered"),
        0.02,
    )

    drift = (r_arousal - 0.5) * 0.35
    overload = load * 0.45
    restore = recovery * 0.5 + (r_safety - 0.5) * 0.2
    arousal_level = _clamp01(prev_arousal + drift - overload + restore)
    stability_index = _clamp01(prev_stability + (r_safety - 0.5) * 0.25 + (attention_level - 0.5) * 0.25 - overload)

    mode = "FOCUSED"
    if attention_level < 0.25 or load > 0.3:
        mode = "FATIGUED"
    elif r_safety < 0.35 or arousal_level > 0.72:
        mode = "CALM"
    elif r_valence < 0.35:
        mode = "RECOVERY"
    adjustments = 0
    if abs(arousal_level - prev_arousal) > 0.02:
        adjustments += 1
    if abs(stability_index - prev_stability) > 0.02:
        adjustments += 1

    return {
        "homeostasis_mode": mode,
        "arousal_level": float(arousal_level),
        "stability_index": float(stability_index),
        "homeostasis_adjustments_count": int(adjustments),
    }


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


__all__ = ["update_homeostasis"]
