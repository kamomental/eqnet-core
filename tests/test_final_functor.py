# -*- coding: utf-8 -*-

from emot_terrain_lab.ops.cleanup_fastpath import summarize_cleanup
from emot_terrain_lab.ops.task_fastpath import summarize_task_fastpath
from emot_terrain_lab.ops.task_profiles import TASK_PROFILES
from emot_terrain_lab.utils.final_functor import (
    AggregateFeature,
    AggregatorKind,
    is_cofinal_subset,
    reduce_feature_stream,
)


def test_is_cofinal_subset_matches_sorted_suffix() -> None:
    assert is_cofinal_subset([1, 2, 3, 4], [2, 4])
    assert not is_cofinal_subset([1, 2, 3], [])
    assert not is_cofinal_subset([1, 2, 3], [2])  # missing >=3


def test_reduce_feature_stream_uses_fast_path_for_union() -> None:
    feature = AggregateFeature("shards", AggregatorKind.UNION)
    value_map = {1: {"a"}, 3: {"b"}}
    value, used_fast = reduce_feature_stream(
        feature,
        timeline_full=[1, 2, 3],
        timeline_subset=[1, 3],
        value_map=value_map,
    )
    assert used_fast is True
    assert value == {"a", "b"}


def test_reduce_feature_stream_falls_back_for_mean() -> None:
    feature = AggregateFeature("dryness", AggregatorKind.MEAN)
    value_map = {1: 0.3, 2: 0.9}
    value, used_fast = reduce_feature_stream(
        feature,
        timeline_full=[1, 2],
        timeline_subset=[1],
        value_map=value_map,
    )
    assert used_fast is False
    assert value == 0.6


def test_summarize_cleanup_uses_new_interface() -> None:
    J = [0.0, 1.0, 2.0, 3.0]
    shards = {0.0: {"s0"}, 1.0: set(), 2.0: {"s1"}, 3.0: set(), 5.0: set(), 10.0: set(), 20.0: set()}
    water = {0.0: 0.0, 1.0: 0.3, 2.0: 0.4, 3.0: 0.8, 5.0: 0.2, 10.0: 0.0, 20.0: 0.0}
    hazard = {0.0: 0.2, 1.0: 0.4, 2.0: 0.6, 3.0: 0.7, 5.0: 0.8, 10.0: 0.9, 20.0: 1.0}
    projector_map = {
        "shards_collected": lambda t, data=shards: data.get(t, set()),
        "danger_zones_handled": lambda t: set(),
        "cleaned_tiles": lambda t: set(),
        "water_absorbed_ml": lambda t, data=water: data.get(t, 0.0),
        "max_hazard_level": lambda t, data=hazard: data.get(t, 0.0),
        "dryness_avg": lambda t: 0.5,
        "smell_ema": lambda t: 0.1,
        "dry_ratio": lambda t: 0.2,
    }
    summary = summarize_cleanup(J, projector_map)
    assert summary["final_ok"] is True
    assert summary["fast"]["shards_collected"] == {"s0", "s1"}
    assert summary["fast"]["water_absorbed_ml"] == 0.6
    assert "dryness_avg" in summary["needs_full"]


def test_rescue_prep_profile_handles_union_and_sum() -> None:
    profile = TASK_PROFILES["rescue_prep"]
    J = [0.0, 1.0, 3.0, 7.0, 15.0]
    projector_map = {
        "aed_fetched": lambda t: {"aed"} if t >= 1.0 else set(),
        "evac_route_cleared": lambda t: {"hall"} if t >= 3.0 else set(),
        "hazard_neutralized": lambda t: {"spill"} if t >= 0.0 else set(),
        "crowd_distance_max": lambda t: t / 10.0,
        "positive_time_spent": lambda t: t,
        "attention_ema": lambda t: 0.5,
        "stress_index_mean": lambda t: 0.4,
        "go_percentile_stream": lambda t: 0.95 if t == 3.0 else 0.7,
        "rarity_stream": lambda t: 0.85 if t == 3.0 else 0.5,
    }
    summary = summarize_task_fastpath(profile, J, projector_map)
    assert summary["final_ok"] is True
    assert summary["fast"]["aed_fetched"] == {"aed"}
    assert "attention_ema" in summary["needs_full"]
    assert summary["predicates"].get("fast_rescue") is True
