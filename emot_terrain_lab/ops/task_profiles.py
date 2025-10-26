# -*- coding: utf-8 -*-
"""Task profiles describing checkpoints and cocontinuous feature specs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..utils.final_functor import AggregatorKind


@dataclass(frozen=True)
class TaskProfile:
    """Describe a fast-path capable scenario."""

    name: str
    checkpoints: Sequence[float]
    features_cocont: Mapping[str, AggregatorKind]
    features_non_cocont: Sequence[str] = field(default_factory=tuple)
    fast_predicates: Optional[Mapping[str, Mapping[str, Any]]] = None
    desc: str = ""
    tags: Sequence[str] = field(default_factory=tuple)


TASK_PROFILES: Dict[str, TaskProfile] = {
    "cleanup": TaskProfile(
        name="cleanup",
        checkpoints=[0.0, 2.0, 5.0, 10.0, 20.0],
        features_cocont={
            "shards_collected": AggregatorKind.UNION,
            "danger_zones_handled": AggregatorKind.UNION,
            "water_absorbed_ml": AggregatorKind.SUM_POS,
            "cleaned_tiles": AggregatorKind.UNION,
            "max_hazard_level": AggregatorKind.SUP,
        },
        features_non_cocont=["dryness_avg", "smell_ema", "dry_ratio"],
        desc="Spill cleanup fast-path; cocontinuous unions/sup/sum_pos supported.",
        tags=("safety", "household"),
    ),
    "rescue_prep": TaskProfile(
        name="rescue_prep",
        checkpoints=[0.0, 1.0, 3.0, 7.0, 15.0],
        features_cocont={
            "aed_fetched": AggregatorKind.UNION,
            "evac_route_cleared": AggregatorKind.UNION,
            "hazard_neutralized": AggregatorKind.UNION,
            "crowd_distance_max": AggregatorKind.SUP,
            "positive_time_spent": AggregatorKind.SUM_POS,
        },
        features_non_cocont=["attention_ema", "stress_index_mean"],
        fast_predicates={
            "fast_rescue": {"type": "go_sc_and_rarity", "pmin": 0.9, "rmin": 0.8},
        },
        desc="Rescue preparation fast-path with AED/route/hazard unions.",
        tags=("safety", "rescue"),
    ),
}


__all__ = ["TaskProfile", "TASK_PROFILES"]
