"""Schema divergence utilities."""
from __future__ import annotations

from typing import Iterable

import numpy as np


def _normalize_prob(vec: np.ndarray) -> np.ndarray:
    total = vec.sum()
    if total <= 0.0:
        return np.full_like(vec, 1.0 / vec.size)
    return vec / total


def _is_distribution(vec: np.ndarray) -> bool:
    if vec.ndim != 1:
        return False
    if np.any(vec < -1e-6):
        return False
    total = vec.sum()
    return np.isclose(total, 1.0, atol=1e-3)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = _normalize_prob(p)
    q = _normalize_prob(q)
    m = 0.5 * (p + q)
    return 0.5 * (_kl_divergence(p, m) + _kl_divergence(q, m))


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-8
    ratio = (p + eps) / (q + eps)
    return float(np.sum(p * np.log(ratio)))


def _cosine_distance(p: np.ndarray, q: np.ndarray) -> float:
    denom = np.linalg.norm(p) * np.linalg.norm(q) + 1e-8
    return float(1.0 - np.dot(p, q) / denom)


def divergence(pred: Iterable[float], post: Iterable[float]) -> float:
    """Auto-select divergence metric (JS for distributions, cosine otherwise)."""
    p = np.asarray(pred, dtype=float)
    q = np.asarray(post, dtype=float)
    if p.shape != q.shape:
        raise ValueError("pred and post must share the same shape")
    if _is_distribution(p) and _is_distribution(q):
        return float(_js_divergence(p, q))
    return _cosine_distance(p, q)


class MetaMonitor:
    """Wrapper to keep API parity with future stateful monitors."""

    def compute(self, prediction: Iterable[float], posterior: Iterable[float]) -> float:
        return divergence(prediction, posterior)
