"""Utilities for computing τ(ρ) relaxation and g(R) synchrony curves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass
class RelaxationResult:
    tau: float
    rho_peak: float
    baseline: float


def relaxation_time(series: Iterable[float], threshold: float = 1.0) -> RelaxationResult:
    """Compute relaxation time τ(ρ) after a perturbation."""
    rho = np.asarray(list(series), dtype=np.float32)
    if rho.size == 0:
        return RelaxationResult(tau=0.0, rho_peak=0.0, baseline=0.0)
    peak_idx = int(np.argmax(rho))
    baseline = float(np.median(rho[-max(1, rho.size // 5) :]))
    decay = np.where(rho[peak_idx:] <= baseline + threshold)[0]
    tau = float(decay[0]) if decay.size else float(rho.size - peak_idx)
    return RelaxationResult(tau=tau, rho_peak=float(rho[peak_idx]), baseline=baseline)


def g_of_r(phases: Iterable[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate g(R): synchrony gain over time window."""
    pairs = np.asarray(list(phases), dtype=np.float32)
    if pairs.size == 0:
        return np.array([]), np.array([])
    r = pairs[:, 0]
    dr = np.gradient(r)
    return r, dr
