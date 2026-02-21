"""Shared safety helpers for associative scoring."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple


def sanitize_weights(
    weights: Dict[str, float],
    *,
    normalize: bool = True,
    fallback_key: str = "semantic",
) -> Dict[str, float]:
    """Clamp negatives and optionally normalize with fallback."""
    safe = {str(k): max(0.0, float(v)) for k, v in (weights or {}).items()}
    if not normalize:
        return safe
    total = sum(safe.values())
    if total <= 1e-9:
        out = {k: 0.0 for k in safe.keys()}
        out[fallback_key] = 1.0
        return out
    return {k: (v / total) for k, v in safe.items()}


def clamp_score(score: float, min_value: float, max_value: float) -> float:
    lo = float(min_value)
    hi = float(max_value)
    if lo > hi:
        lo, hi = hi, lo
    val = float(score)
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val


def calc_saturation_stats(
    scores: Iterable[float],
    min_value: float,
    max_value: float,
    *,
    eps: float = 1e-9,
) -> Tuple[int, int, int]:
    """Return (sat_min_count, sat_max_count, n)."""
    lo = float(min_value)
    hi = float(max_value)
    if lo > hi:
        lo, hi = hi, lo
    sat_min = 0
    sat_max = 0
    n = 0
    for raw in scores:
        n += 1
        val = float(raw)
        if abs(val - lo) <= eps:
            sat_min += 1
        if abs(val - hi) <= eps:
            sat_max += 1
    return sat_min, sat_max, n
