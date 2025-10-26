"""Lightweight Hawkes process fitting for Wave P3."""

from __future__ import annotations

import numpy as np


def exp_kernel_conv(events: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Compute exponential kernel convolution for binary event matrix."""
    if events.ndim != 2:
        raise ValueError("events must be [T, N].")
    T, N = events.shape
    conv = np.zeros_like(events, dtype=float)
    decay = float(np.exp(-beta))
    scale = float(beta)
    for n in range(N):
        acc = 0.0
        for t in range(T):
            acc = decay * acc + scale * events[t, n]
            conv[t, n] = acc
    return conv


def fit_hawkes(
    events: np.ndarray,
    beta: float = 1.0,
    l2: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit μ and α coefficients via regularised least squares."""
    if events.ndim != 2:
        raise ValueError("events must be [T, N].")
    T, N = events.shape
    H = exp_kernel_conv(events, beta)
    X = np.concatenate([np.ones((T, 1)), H], axis=1)
    Y = events.astype(float)
    XtX = X.T @ X + l2 * np.eye(X.shape[1])
    XtY = X.T @ Y
    W = np.linalg.solve(XtX, XtY)
    mu = W[0]
    A = W[1:].T
    return mu, A


__all__ = ["exp_kernel_conv", "fit_hawkes"]
