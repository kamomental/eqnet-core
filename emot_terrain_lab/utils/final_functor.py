# -*- coding: utf-8 -*-
"""Helpers for cocontinuous aggregations and cofinal fast-path reductions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


class AggregatorKind(str, Enum):
    """Kinds of aggregations supported by the fast-path planner."""

    UNION = "union"
    SUP = "sup"
    SUM_POS = "sum_pos"
    MEAN = "mean"
    EMA = "ema"
    QUANTILE = "quantile"
    RATIO = "ratio"
    TOPK = "topk"


COCONTINUOUS_KINDS = {AggregatorKind.UNION, AggregatorKind.SUP, AggregatorKind.SUM_POS}


@dataclass(frozen=True)
class AggregateFeature:
    """Feature metadata describing how it can be aggregated."""

    name: str
    kind: AggregatorKind
    domain: Optional[str] = None
    description: Optional[str] = None


class AggregateRegistry:
    """Registry for aggregate-aware features."""

    def __init__(self, features: Optional[Iterable[AggregateFeature]] = None) -> None:
        self._features: Dict[str, AggregateFeature] = {}
        if features:
            for feature in features:
                self.register(feature)

    def register(self, feature: AggregateFeature) -> None:
        self._features[feature.name] = feature

    def get(self, name: str) -> Optional[AggregateFeature]:
        return self._features.get(name)

    def features(self) -> List[AggregateFeature]:
        return list(self._features.values())

    def __contains__(self, name: str) -> bool:
        return name in self._features


def is_cocontinuous(kind: AggregatorKind) -> bool:
    return kind in COCONTINUOUS_KINDS


def is_cofinal_subset(full_order: Sequence[Any], subset: Sequence[Any]) -> bool:
    """
    Simple cofinality check for totally-ordered index sets (e.g. timestamps).

    Returns True when every element in full_order has a greater-or-equal element in subset.
    """
    if not full_order or not subset:
        return False
    full_sorted = _sorted_unique(full_order)
    subset_sorted = _sorted_unique(subset)
    idx = 0
    limit = len(subset_sorted)
    for value in full_sorted:
        while idx < limit and subset_sorted[idx] < value:
            idx += 1
        if idx >= limit:
            return False
    return True


def is_cofinal_subset_poset(subset: Sequence[Any], full_order: Sequence[Any]) -> bool:
    """Alias that follows the usual notation I⊆J (subset first, then full)."""
    return is_cofinal_subset(full_order, subset)


def reduce_feature_stream(
    feature: AggregateFeature,
    *,
    timeline_full: Sequence[Any],
    timeline_subset: Sequence[Any],
    value_map: Mapping[Any, Any],
) -> tuple[Any, bool]:
    """
    Reduce a feature stream, using the subset timeline when it is cofinal and cocontinuous.

    Returns (value, used_fast_path_flag).
    """
    aggregator = _AGGREGATORS.get(feature.kind)
    if aggregator is None:
        raise ValueError(f"unsupported aggregator kind: {feature.kind}")

    # Collect the full sequence (to ensure we have data even if fast-path fails)
    values_full = [value_map[key] for key in timeline_full if key in value_map]
    if not values_full:
        return None, False

    can_fast_path = is_cocontinuous(feature.kind) and is_cofinal_subset(timeline_full, timeline_subset)
    if can_fast_path:
        values_subset = [value_map[key] for key in timeline_subset if key in value_map]
        if values_subset:
            return aggregator(values_subset), True

    return aggregator(values_full), False


def _agg_union(values: Iterable[Any]) -> set[Any]:
    result: set[Any] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (set, frozenset)):
            result.update(value)
        elif isinstance(value, (list, tuple)):
            result.update(item for item in value if item is not None)
        elif isinstance(value, Mapping):
            result.update(key for key, present in value.items() if present)
        else:
            result.add(value)
    return result


def _agg_sup(values: Iterable[Any]) -> Any:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return max(filtered)


def _agg_sum_pos(values: Iterable[Any]) -> float:
    total = 0.0
    for value in values:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0.0:
            total += numeric
    return total


def _agg_mean(values: Iterable[Any]) -> Optional[float]:
    total = 0.0
    count = 0
    for value in values:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        total += numeric
        count += 1
    if not count:
        return None
    return total / count


_AGGREGATORS: Dict[AggregatorKind, Callable[[Iterable[Any]], Any]] = {
    AggregatorKind.UNION: _agg_union,
    AggregatorKind.SUP: _agg_sup,
    AggregatorKind.SUM_POS: _agg_sum_pos,
    AggregatorKind.MEAN: _agg_mean,
}


def colim_over_index(
    kind: AggregatorKind | str,
    index_points: Sequence[Any],
    projector: Callable[[Any], Any],
) -> Any:
    """Aggregate projector(index) across the sequence using the requested cocontinuous operation."""
    agg_kind = kind
    if isinstance(agg_kind, str):
        try:
            agg_kind = AggregatorKind(agg_kind)
        except ValueError as exc:
            raise ValueError(f"unsupported aggregator kind: {kind}") from exc
    aggregator = _AGGREGATORS.get(agg_kind)
    if aggregator is None:
        raise ValueError(f"aggregator {agg_kind} not implemented")
    values = (projector(point) for point in index_points)
    return aggregator(values)


def _sorted_unique(values: Sequence[Any]) -> List[Any]:
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    try:
        ordered.sort()
    except TypeError:
        # Fall back to insertion order if values are not comparable
        pass
    return ordered


__all__ = [
    "AggregateFeature",
    "AggregateRegistry",
    "AggregatorKind",
    "COCONTINUOUS_KINDS",
    "is_cocontinuous",
    "is_cofinal_subset",
    "is_cofinal_subset_poset",
    "reduce_feature_stream",
    "colim_over_index",
]
