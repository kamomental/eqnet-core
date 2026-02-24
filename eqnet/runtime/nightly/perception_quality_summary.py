from __future__ import annotations

from typing import Any, Dict, List, Mapping

from eqnet.runtime.future_contracts import (
    compute_perception_quality as _compute_perception_quality,
    summarize_perception_quality as _summarize_perception_quality,
    summarize_perception_quality_breakdown as _summarize_perception_quality_breakdown,
)


def compute_perception_quality(
    event: Mapping[str, Any],
    *,
    now_ts_ms: int,
    rules: Mapping[str, Any],
) -> Dict[str, Any]:
    return _compute_perception_quality(event, now_ts_ms=now_ts_ms, rules=rules)


def summarize_perception_quality(
    events: List[Mapping[str, Any]],
    *,
    now_ts_ms: int,
    rules: Mapping[str, Any],
) -> Dict[str, Any]:
    return _summarize_perception_quality(events, now_ts_ms=now_ts_ms, rules=rules)


def summarize_perception_quality_breakdown(
    events: List[Mapping[str, Any]],
    *,
    now_ts_ms: int,
    rules: Mapping[str, Any],
) -> Dict[str, Any]:
    return _summarize_perception_quality_breakdown(events, now_ts_ms=now_ts_ms, rules=rules)

