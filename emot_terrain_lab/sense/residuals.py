# -*- coding: utf-8 -*-
"""Residual calculator between true sense features and language reconstruction."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def compute_residual(
    f: Dict[str, float],
    fhat: Dict[str, float],
    shareability: Dict[str, float],
    weights: Dict[str, float],
    *,
    topk: int = 3,
) -> Dict[str, object]:
    """Return scalar delta and per-dimension residual after shareability weighting."""

    keys = sorted(set(f) & set(fhat) & set(shareability))
    if not keys:
        return {"delta": 0.0, "residual": {}, "top": []}

    x = np.array([float(f[k]) for k in keys], dtype=float)
    xh = np.array([float(fhat[k]) for k in keys], dtype=float)
    s = np.array([float(shareability.get(k, 0.5)) for k in keys], dtype=float)
    w = np.array([float(weights.get(k, 1.0)) for k in keys], dtype=float)

    residual = s * np.abs(x - xh)
    denom = float((s * w).sum())
    if denom <= 0.0:
        delta = float(residual.mean())
    else:
        delta = float((residual * w).sum() / denom)

    ordering = np.argsort(-(residual * w))
    top_pairs: List[Tuple[str, float]] = [
        (keys[idx], float(residual[idx])) for idx in ordering[:topk]
    ]
    return {
        "delta": max(0.0, min(1.0, delta)),
        "residual": {name: float(residual[i]) for i, name in enumerate(keys)},
        "top": top_pairs,
    }


__all__ = ["compute_residual"]
