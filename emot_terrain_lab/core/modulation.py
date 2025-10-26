# -*- coding: utf-8 -*-
"""Utility for applying coupling matrices to control dictionaries."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple


def apply_coupling(
    base: Mapping[str, float | int],
    signals: Mapping[str, float],
    coupling: Mapping[str, Mapping[str, float]],
) -> Dict[str, float | int]:
    """Deprecated signature retained for backward compatibility."""
    updated, _ = apply_coupling_with_limits(base, signals, coupling)
    return updated


def apply_coupling_with_limits(
    base: Mapping[str, float | int],
    signals: Mapping[str, float],
    coupling: Mapping[str, Mapping[str, float]],
    *,
    limits: Optional[Mapping[str, Tuple[float, float]]] = None,
    d_tau: float = 0.7,
) -> Tuple[Dict[str, float | int], Dict[str, float]]:
    """Combine base controls with signal-driven adjustments."""
    updated: Dict[str, float | int] = dict(base)
    contributions: Dict[str, float] = {}
    tau_scale = min(1.0, 0.7 / max(1e-6, float(d_tau)))
    for src, val in signals.items():
        if src not in coupling:
            continue
        for dst, gain in coupling[src].items():
            delta = float(gain) * float(val) * tau_scale
            current = float(updated.get(dst, 0.0)) + delta
            updated[dst] = current
            contributions[dst] = round(contributions.get(dst, 0.0) + delta, 6)
    _clip_controls(updated, limits)
    return updated, contributions


def _clip_controls(
    controls: Dict[str, float | int],
    limits: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> None:
    """Clamp known control dimensions to safe ranges."""
    if limits:
        for key, bounds in limits.items():
            if key in controls and isinstance(bounds, (tuple, list)) and len(bounds) == 2:
                lo, hi = float(bounds[0]), float(bounds[1])
                controls[key] = max(lo, min(hi, float(controls[key])))
    if "directness" in controls:
        controls["directness"] = max(-0.2, min(0.2, float(controls["directness"])))
    if "pause_ms" in controls:
        controls["pause_ms"] = int(max(0, min(400, float(controls["pause_ms"]))))
    if "temp_mul" in controls:
        controls["temp_mul"] = max(0.6, min(1.2, float(controls["temp_mul"])))
    if "top_p_mul" in controls:
        controls["top_p_mul"] = max(0.5, min(1.5, float(controls["top_p_mul"])))


__all__ = ["apply_coupling", "apply_coupling_with_limits"]
