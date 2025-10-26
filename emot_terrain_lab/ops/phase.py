"""Phase extraction utilities for Sigma/Psi signals."""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


def phase_from_series(series: np.ndarray) -> float:
    """Return instantaneous phase from the latest point in the series."""
    if series.ndim != 1:
        raise ValueError("series must be 1D.")
    if series.size < 8:
        return 0.0
    centred = series - np.mean(series)
    analytic = hilbert(centred)
    return float(np.mod(np.angle(analytic[-1]), 2 * np.pi))


__all__ = ["phase_from_series"]
