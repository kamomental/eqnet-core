# -*- coding: utf-8 -*-
"""Mood-driven control modulation utilities."""

from __future__ import annotations

from typing import Any, Dict


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def mood_controls(
    base: Dict[str, Any],
    mood: Dict[str, float],
    heartiness: float,
    *,
    style: str,
) -> Dict[str, Any]:
    """Blend mood metrics into control signals with small gains."""
    out = dict(base)
    f = float(_clip(heartiness, 0.0, 1.0))
    uncertainty = float(mood.get("u", mood.get("uncertainty", 0.3)))
    arousal = float(mood.get("a", mood.get("arousal", 0.5)))
    direct_drive = float(mood.get("d", mood.get("dominance", 0.0)))

    out["temp_mul"] = _clip(1.0 - 0.16 * f * (0.5 + uncertainty), 0.6, 1.2)
    out["top_p_mul"] = _clip(1.0 - 0.12 * f * (0.5 + uncertainty), 0.6, 1.1)
    out["pause_ms"] = int(
        _clip(250.0 * f * (0.5 + max(0.0, 0.5 - arousal)), 0.0, 300.0)
    )
    out["directness"] = _clip(0.10 * f * (0.5 + direct_drive), -0.1, 0.2)

    if style == "tidy_strict":
        out["pause_ms"] = min(out["pause_ms"], 80)
    elif style == "tidy_humming":
        out["pause_ms"] += 70
    return out


__all__ = ["mood_controls"]
