# -*- coding: utf-8 -*-
"""Helpers for aggregating sharedness / language loss telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class SharednessSample:
    delta: float
    language_loss: float


def summarise(samples: Iterable[SharednessSample]) -> dict:
    """Compute basic stats for Nightly dashboards."""

    deltas: List[float] = []
    losses: List[float] = []
    for sample in samples:
        deltas.append(float(sample.delta))
        losses.append(float(sample.language_loss))
    if not deltas:
        return {
            "count": 0,
            "delta_avg": 0.0,
            "delta_p95": 0.0,
            "language_loss_avg": 0.0,
            "language_loss_p95": 0.0,
        }

    deltas.sort()
    losses.sort()
    idx95 = max(0, int(0.95 * (len(deltas) - 1)))
    return {
        "count": len(deltas),
        "delta_avg": sum(deltas) / len(deltas),
        "delta_p95": deltas[idx95],
        "language_loss_avg": sum(losses) / len(losses),
        "language_loss_p95": losses[idx95],
    }


__all__ = ["SharednessSample", "summarise"]
