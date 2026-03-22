from inner_os.social_topology_state import (
    coerce_social_topology_label,
    derive_social_topology_state,
)


def test_social_topology_state_prefers_threaded_group_for_multiple_close_threads() -> None:
    state = derive_social_topology_state(
        scene_state={
            "social_topology": "group_present",
            "privacy_level": 0.46,
            "norm_pressure": 0.22,
            "environmental_load": 0.18,
            "scene_tags": ["task:coordination"],
        },
        relation_competition_state={
            "state": "competing_threads",
            "competition_level": 0.52,
            "winner_margin": 0.08,
            "total_people": 3,
            "dominant_person_id": "person:harbor",
        },
        related_person_ids=["person:harbor", "person:friend", "person:guest"],
        self_state={},
    ).to_dict()

    assert state["state"] == "threaded_group"
    assert state["threading_pressure"] >= state["visibility_pressure"]
    assert "multiple_people" in state["dominant_inputs"]
    assert coerce_social_topology_label(state["state"]) == "multi_person"


def test_social_topology_state_prefers_public_visible_for_low_privacy_scene() -> None:
    state = derive_social_topology_state(
        scene_state={
            "social_topology": "public_visible",
            "privacy_level": 0.14,
            "norm_pressure": 0.62,
            "environmental_load": 0.28,
            "scene_tags": ["socially_exposed", "high_norm"],
        },
        relation_competition_state={
            "state": "single_anchor",
            "competition_level": 0.0,
            "winner_margin": 0.36,
            "total_people": 1,
            "dominant_person_id": "person:user",
        },
        related_person_ids=["person:user"],
        self_state={},
    ).to_dict()

    assert state["state"] == "public_visible"
    assert state["visibility_pressure"] > state["threading_pressure"]
    assert "public_visibility" in state["dominant_inputs"]
    assert coerce_social_topology_label(state["state"]) == "public_visible"
