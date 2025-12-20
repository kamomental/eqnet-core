"""Precision-weighted query pressure computation."""
from __future__ import annotations

from typing import Iterable, Union

import numpy as np

ArrayLike = Union[float, Iterable[float], np.ndarray]


class QueryEngine:
    """Compute u_t = ||Pi_t * epsilon_t||."""

    def compute(self, epsilon: ArrayLike, precision: ArrayLike) -> float:
        eps = np.asarray(epsilon, dtype=float)
        Pi = precision
        if np.isscalar(Pi):
            weighted = float(Pi) * eps
        else:
            matrix = np.asarray(Pi, dtype=float)
            if matrix.ndim == 1:
                weighted = matrix * eps
            elif matrix.ndim == 2:
                weighted = matrix @ eps
            else:
                raise ValueError("precision must be scalar, vector, or matrix")
        return float(np.linalg.norm(weighted))
