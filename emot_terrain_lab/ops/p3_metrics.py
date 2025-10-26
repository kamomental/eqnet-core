"""Metrics utilities for Wave P3 multi-agent experiments."""

from __future__ import annotations

import numpy as np


def propagation_rate(
    events_i: np.ndarray,
    events_j: np.ndarray,
    tau_sec: float,
    fps: float = 1.0,
) -> float:
    """Estimate conditional propagation P(bud_j | bud_i, Δt ≤ τ)."""
    if events_i.shape != events_j.shape:
        raise ValueError("events_i and events_j must have identical shape.")
    total = int(np.sum(events_i > 0))
    if total == 0:
        return 0.0
    window = int(max(1, round(tau_sec * fps)))
    hits = 0
    indices = np.flatnonzero(events_i > 0)
    for idx in indices:
        lo = idx
        hi = min(events_i.shape[0], idx + window + 1)
        if np.any(events_j[lo:hi] > 0):
            hits += 1
    return hits / total


def kuramoto_R(theta: np.ndarray) -> float:
    """Compute Kuramoto order parameter from phase array theta."""
    if theta.ndim != 1:
        raise ValueError("theta must be 1D.")
    if theta.size == 0:
        return 0.0
    z = np.exp(1j * theta)
    return float(np.abs(np.mean(z)))


__all__ = ["propagation_rate", "kuramoto_R"]
