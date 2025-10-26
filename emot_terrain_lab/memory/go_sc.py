# -*- coding: utf-8 -*-
"""Helpers for GO-SC scoring and percentile tracking."""

from __future__ import annotations

from bisect import bisect_left, insort
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Mapping, MutableMapping


Numeric = float | int


@dataclass(frozen=True)
class GoSCWeights:
    """Weight definition for GO-SC scoring."""

    weights: Mapping[str, float]
    percentile_window: int = 2048


class StreamingPercentiler:
    """Approximate percentile tracker over a sliding window."""

    def __init__(self, window: int) -> None:
        self.window = max(int(window or 0), 2)
        self._sorted: list[float] = []
        self._queue: Deque[float] = deque()

    def update_and_percentile(self, value: float) -> float:
        """Insert value, return rank/(N-1) with small-sample guard."""
        count = len(self._sorted)
        if count < 2:
            percentile = 0.5
        else:
            pos = bisect_left(self._sorted, value)
            percentile = pos / max(1, count - 1)
            percentile = max(0.0, min(1.0, percentile))

        insort(self._sorted, value)
        self._queue.append(value)
        if len(self._queue) > self.window:
            old = self._queue.popleft()
            idx = bisect_left(self._sorted, old)
            if idx < len(self._sorted) and self._sorted[idx] == old:
                self._sorted.pop(idx)
        return percentile


def weighted_score(weights: Mapping[str, float], features: Mapping[str, Numeric]) -> float:
    """Simple linear combiner; missing keys default to zero."""
    result = 0.0
    keys = set(weights.keys()) | set(features.keys())
    for key in keys:
        try:
            w = float(weights.get(key, 0.0))
            v = float(features.get(key, 0.0))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        result += w * v
    return result


def merge_feature_sources(base: MutableMapping[str, float], extras: Mapping[str, Any]) -> None:
    """Add numeric extras into base map in-place."""
    for key, value in extras.items():
        if isinstance(value, (int, float)):
            base[key] = float(value)
            continue
        try:
            base[key] = float(value)
        except (TypeError, ValueError):
            continue


def extract_features(event: Mapping[str, Any]) -> Dict[str, float]:
    """Derive a minimal feature vector from an event/trace structure."""
    features: Dict[str, float] = {}
    meta: Mapping[str, Any] = event.get("meta", {}) or {}
    value: Mapping[str, Any] = event.get("value", {}) or {}

    merge_feature_sources(
        features,
        {
            "uncertainty": event.get("uncertainty", 0.0),
            "value_total": value.get("total", 0.0),
            "novelty": meta.get("novelty", meta.get("novelty_score", 0.0)),
            "surprise": meta.get("surprise", meta.get("surprise_score", 0.0)),
            "rarity": meta.get("rarity", meta.get("salience", 0.0)),
            "schema_gain": meta.get("schema_gain", 0.0),
            "risk": meta.get("norm_risk", 0.0),
        },
    )

    go_feats = meta.get("go_features")
    if isinstance(go_feats, Mapping):
        merge_feature_sources(features, go_feats)
    return features


def compute_go_sc(
    cfg: GoSCWeights,
    event: Mapping[str, Any],
    percentiler: StreamingPercentiler,
) -> tuple[float, float]:
    """Return (score, percentile) for the given event."""
    features = extract_features(event)
    score = weighted_score(cfg.weights, features)
    percentile = percentiler.update_and_percentile(score)
    return score, percentile


__all__ = [
    "GoSCWeights",
    "StreamingPercentiler",
    "compute_go_sc",
    "extract_features",
    "weighted_score",
]
