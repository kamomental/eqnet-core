from inner_os.agenda_window_state import derive_agenda_window_state


def test_public_repair_waits_for_private_window() -> None:
    state = derive_agenda_window_state(
        self_state={
            "stress": 0.18,
            "recovery_need": 0.12,
            "prospective_memory_pull": 0.34,
        },
        agenda_state={"state": "repair", "reason": "repair_window", "score": 0.62},
        body_recovery_guard={"state": "open", "score": 0.18},
        body_homeostasis_state={"state": "steady", "score": 0.12},
        homeostasis_budget_state={"state": "steady", "score": 0.1},
        initiative_followup_bias={"state": "reopen_softly", "score": 0.42},
        commitment_state={"target": "repair", "score": 0.66},
        relational_continuity_state={"state": "reopening", "score": 0.54},
        relation_competition_state={"state": "single_relation"},
        social_topology_state={
            "state": "public_visible",
            "score": 0.72,
            "visibility_pressure": 0.68,
            "threading_pressure": 0.16,
            "hierarchy_pressure": 0.18,
        },
        cultural_conversation_state={"state": "public_courteous", "directness_ceiling": 0.32, "politeness_pressure": 0.62},
        related_person_ids=["user"],
        relation_bias_strength=0.64,
        scene_family="reverent_distance",
    ).to_dict()

    assert state["state"] == "next_private_window"
    assert state["carry_target"] == "same_person_private_window"
    assert state["deferral_budget"] >= 0.28


def test_threaded_group_waits_for_same_group_window() -> None:
    state = derive_agenda_window_state(
        self_state={
            "semantic_seed_strength": 0.28,
            "semantic_seed_recurrence": 0.4,
            "prospective_memory_pull": 0.26,
        },
        agenda_state={"state": "revisit", "reason": "revisit_open_loop", "score": 0.58},
        body_recovery_guard={"state": "open", "score": 0.08},
        body_homeostasis_state={"state": "steady", "score": 0.08},
        homeostasis_budget_state={"state": "steady", "score": 0.08},
        initiative_followup_bias={"state": "reopen_softly", "score": 0.38},
        commitment_state={"target": "hold", "score": 0.26},
        relational_continuity_state={"state": "holding_thread", "score": 0.52},
        relation_competition_state={"state": "multi_relation"},
        social_topology_state={
            "state": "threaded_group",
            "score": 0.7,
            "visibility_pressure": 0.22,
            "threading_pressure": 0.74,
            "hierarchy_pressure": 0.08,
        },
        cultural_conversation_state={"state": "group_attuned", "directness_ceiling": 0.44, "politeness_pressure": 0.34},
        related_person_ids=["user", "friend"],
        relation_bias_strength=0.5,
        scene_family="shared_world",
    ).to_dict()

    assert state["state"] == "next_same_group_window"
    assert state["carry_target"] == "same_group_thread"


def test_recurrent_semantic_seed_allows_opportunistic_reentry() -> None:
    state = derive_agenda_window_state(
        self_state={
            "semantic_seed_strength": 0.62,
            "semantic_seed_recurrence": 1.6,
            "prospective_memory_pull": 0.42,
            "pending_meaning": 0.36,
            "agenda_window_focus": "opportunistic_reentry",
            "agenda_window_bias": 0.18,
        },
        agenda_state={"state": "revisit", "reason": "insight_revisit", "score": 0.54},
        body_recovery_guard={"state": "open", "score": 0.06},
        body_homeostasis_state={"state": "steady", "score": 0.08},
        homeostasis_budget_state={"state": "steady", "score": 0.1},
        initiative_followup_bias={"state": "reopen_softly", "score": 0.4},
        commitment_state={"target": "hold", "score": 0.22},
        relational_continuity_state={"state": "reopening", "score": 0.46},
        relation_competition_state={"state": "ambient"},
        social_topology_state={
            "state": "ambient",
            "score": 0.34,
            "visibility_pressure": 0.06,
            "threading_pressure": 0.04,
            "hierarchy_pressure": 0.04,
        },
        cultural_conversation_state={"state": "group_attuned", "directness_ceiling": 0.4, "politeness_pressure": 0.2},
        related_person_ids=[],
        relation_bias_strength=0.08,
        scene_family="shared_world",
    ).to_dict()

    assert state["state"] in {"opportunistic_reentry", "next_same_culture_window"}
    assert state["opportunistic_ok"] is True


def test_temporal_reentry_bias_weakly_lifts_private_reentry_window() -> None:
    base_kwargs = {
        "agenda_state": {"state": "repair", "reason": "repair_window", "score": 0.52},
        "body_recovery_guard": {"state": "open", "score": 0.12},
        "body_homeostasis_state": {"state": "steady", "score": 0.08},
        "homeostasis_budget_state": {"state": "steady", "score": 0.08},
        "initiative_followup_bias": {"state": "reopen_softly", "score": 0.44},
        "commitment_state": {"target": "repair", "score": 0.48},
        "relational_continuity_state": {"state": "reopening", "score": 0.5},
        "relation_competition_state": {"state": "single_relation"},
        "social_topology_state": {
            "state": "public_visible",
            "score": 0.64,
            "visibility_pressure": 0.58,
            "threading_pressure": 0.12,
            "hierarchy_pressure": 0.12,
        },
        "cultural_conversation_state": {
            "state": "public_courteous",
            "directness_ceiling": 0.32,
            "politeness_pressure": 0.56,
        },
        "related_person_ids": ["user"],
        "relation_bias_strength": 0.52,
        "scene_family": "reverent_distance",
    }
    without_bias = derive_agenda_window_state(
        self_state={
            "stress": 0.16,
            "recovery_need": 0.12,
            "prospective_memory_pull": 0.22,
        },
        **base_kwargs,
    ).to_dict()
    with_bias = derive_agenda_window_state(
        self_state={
            "stress": 0.16,
            "recovery_need": 0.12,
            "prospective_memory_pull": 0.22,
            "temporal_reentry_pull": 0.62,
            "temporal_relation_reentry_pull": 0.48,
            "temporal_continuity_pressure": 0.42,
            "temporal_timeline_coherence": 0.38,
            "temporal_membrane_mode": "reentry",
        },
        **base_kwargs,
    ).to_dict()

    assert (
        with_bias["scores"]["next_private_window"]
        > without_bias["scores"]["next_private_window"]
    )
    assert "temporal_reentry_pull" in with_bias["dominant_inputs"]
