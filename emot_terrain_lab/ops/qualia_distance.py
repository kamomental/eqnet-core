"""Qualia field distance approximations for P3/P4 experiments."""

from __future__ import annotations

import numpy as np


def qualia_distance_cos(field_i: np.ndarray, field_j: np.ndarray) -> float:
    """Compute cosine-distance proxy between two qualia fields."""
    if field_i.shape != field_j.shape:
        raise ValueError("Qualia fields must be same shape.")
    vi = field_i.ravel().astype(float)
    vj = field_j.ravel().astype(float)
    ni = float(np.linalg.norm(vi) + 1e-9)
    nj = float(np.linalg.norm(vj) + 1e-9)
    cos = float(np.dot(vi, vj) / (ni * nj))
    return float(1.0 - cos)


__all__ = ["qualia_distance_cos"]
