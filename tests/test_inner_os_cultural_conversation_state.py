from inner_os.cultural_conversation_state import derive_cultural_conversation_state


def test_cultural_conversation_prefers_public_courteous_for_ja_public_scene() -> None:
    state = derive_cultural_conversation_state(
        self_state={
            "culture_id": "ja-JP",
            "community_id": "harbor_collective",
            "culture_resonance": 0.42,
            "related_person_id": "user",
        },
        social_topology_state={
            "state": "public_visible",
            "visibility_pressure": 0.74,
            "threading_pressure": 0.18,
            "hierarchy_pressure": 0.12,
        },
        relation_competition_state={"dominant_person_id": "user", "total_people": 3},
        relational_style_memory_state={
            "state": "warm_companion",
            "playful_ceiling": 0.38,
            "advice_tolerance": 0.54,
            "lexical_familiarity": 0.44,
            "warmth_bias": 0.48,
        },
        body_recovery_guard={"state": "open", "score": 0.08},
    ).to_dict()

    assert state["state"] in {"public_courteous", "careful_polite"}
    assert state["joke_ratio_ceiling"] <= 0.45
    assert state["politeness_pressure"] >= 0.24


def test_cultural_conversation_prefers_group_attuned_for_en_threaded_group() -> None:
    state = derive_cultural_conversation_state(
        self_state={
            "culture_id": "en-US",
            "community_id": "makerspace",
            "culture_resonance": 0.34,
            "related_person_id": "friend",
        },
        social_topology_state={
            "state": "threaded_group",
            "visibility_pressure": 0.22,
            "threading_pressure": 0.68,
            "hierarchy_pressure": 0.08,
        },
        relation_competition_state={"dominant_person_id": "friend", "total_people": 4},
        relational_style_memory_state={
            "state": "light_playful",
            "playful_ceiling": 0.66,
            "advice_tolerance": 0.52,
            "lexical_familiarity": 0.58,
            "warmth_bias": 0.44,
        },
        body_recovery_guard={"state": "open", "score": 0.08},
    ).to_dict()

    assert state["state"] in {"group_attuned", "casual_shared"}
    assert state["joke_ratio_ceiling"] >= 0.34
    assert state["group_attunement"] >= 0.24
