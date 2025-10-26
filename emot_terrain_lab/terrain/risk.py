# -*- coding: utf-8 -*-
"""
Risk and stress indicator helpers (e.g., SVaR percentiles) built on top of the
field metrics the system already collects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PercentileConfig:
    metrics: Sequence[str]
    percentiles: Sequence[float]


def _clean_series(records: Iterable[Mapping], key: str) -> np.ndarray:
    values: List[float] = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        try:
            values.append(abs(float(value)))
        except (TypeError, ValueError):
            continue
    return np.asarray(values, dtype=float)


def compute_percentiles(
    records: Sequence[Mapping],
    config: PercentileConfig,
) -> Dict[str, Dict[str, float]]:
    """
    Compute absolute-value percentiles for the provided metric keys.
    Returns a dict like {"entropy": {"p90": 3.1, "p95": 3.5, ...}, ...}.
    """
    output: Dict[str, Dict[str, float]] = {}
    if not records:
        return output

    for key in config.metrics:
        series = _clean_series(records, key)
        if series.size == 0:
            continue
        thresholds: Dict[str, float] = {}
        for pct in config.percentiles:
            pct_clamped = max(0.0, min(100.0, pct))
            thresholds[f"p{int(pct_clamped)}"] = float(np.percentile(series, pct_clamped))
        output[key] = thresholds
    return output


def add_risk_flags(
    rows: Sequence[Mapping],
    percentile_summary: Mapping[str, Mapping[str, float]],
    high_watermark: str = "p95",
) -> List[Dict]:
    """
    Add per-row boolean flags (0/1) indicating whether the absolute metric value
    exceeds the configured percentile (defaults to the 95th percentile).
    Returns a new list of dicts, leaving the input untouched.
    """
    annotated: List[Dict] = []
    for row in rows:
        enriched = dict(row)
        for metric, percentiles in percentile_summary.items():
            thresh = percentiles.get(high_watermark)
            if thresh is None:
                continue
            value = row.get(metric)
            try:
                exceeds = 1 if abs(float(value)) >= float(thresh) else 0
            except (TypeError, ValueError):
                exceeds = 0
            enriched[f"{metric}_risk_{high_watermark}"] = exceeds
        annotated.append(enriched)
    return annotated


def latest_snapshot(
    rows: Sequence[Mapping],
    percentile_summary: Mapping[str, Mapping[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Prepare a snapshot comparing the most recent readings with percentile
    thresholds so dashboards can readily highlight risk overshoots.
    """
    if not rows:
        return {}
    latest = rows[-1]
    snapshot: Dict[str, Dict[str, float]] = {}
    for metric, percentiles in percentile_summary.items():
        value = latest.get(metric)
        try:
            current = float(value)
        except (TypeError, ValueError):
            continue
        snapshot[metric] = {
            "current": current,
            **{label: float(threshold) for label, threshold in percentiles.items()},
        }
    return snapshot

