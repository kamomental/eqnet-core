# -*- coding: utf-8 -*-

from emot_terrain_lab.ops.task_profiles import TASK_PROFILES
from emot_terrain_lab.ops.task_fastpath import summarize_task_fastpath


def test_profiles_fastpath_minimal() -> None:
    J = [0, 1, 2, 3, 7, 9, 10]
    shards = {0: set(), 1: set(), 2: {"a"}, 3: set(), 7: {"b"}, 9: set(), 10: {"c"}}
    water = {0: 0.0, 1: 0.2, 2: 0.3, 3: 0.0, 7: 0.5, 9: 0.0, 10: 0.7}
    hazard = {0: 0.1, 1: 0.2, 2: 0.6, 3: 0.4, 7: 0.9, 9: 0.3, 10: 0.8}
    go_p = {0: 0.7, 1: 0.8, 2: 0.92, 3: 0.4, 7: 0.88, 9: 0.6, 10: 0.5}
    rarity = {0: 0.9, 1: 0.7, 2: 0.85, 3: 0.3, 7: 0.92, 9: 0.4, 10: 0.2}

    projector_map = {
        "shards_collected": lambda t: shards[t],
        "cleaned_tiles": lambda t: set(),
        "danger_zones_handled": lambda t: set(),
        "water_absorbed_ml": lambda t: water[t],
        "max_hazard_level": lambda t: hazard[t],
        "aed_fetched": lambda t: set(),
        "evac_route_cleared": lambda t: set(),
        "hazard_neutralized": lambda t: set(),
        "crowd_distance_max": lambda t: hazard[t],
        "positive_time_spent": lambda t: water[t],
        "go_percentile_stream": lambda t: go_p[t],
        "rarity_stream": lambda t: rarity[t],
    }

    for name in ("cleanup", "rescue_prep"):
        profile = TASK_PROFILES[name]
        summary = summarize_task_fastpath(profile, J, projector_map)
        assert summary["final_ok"] is True
        assert set(summary["fast"].keys()) == set(profile.features_cocont.keys())
        assert set(summary["needs_full"]) == set(profile.features_non_cocont)
