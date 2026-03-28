from inner_os.situation_risk_state import derive_situation_risk_state


def test_situation_risk_state_distinguishes_routine_tool_context_from_public_threat() -> None:
    routine = derive_situation_risk_state(
        current_risks=["sharp_tool_visible"],
        scene_state={
            "place_mode": "kitchen",
            "privacy_level": 0.82,
            "social_topology": "one_to_one",
            "task_phase": "shared_task",
            "scene_family": "shared_world",
            "safety_margin": 0.74,
            "mobility_context": "stationary",
            "scene_tags": ["task:shared_task", "private"],
        },
        self_state={
            "trust_memory": 0.74,
            "familiarity": 0.7,
            "attachment": 0.62,
            "continuity_score": 0.68,
        },
    ).to_dict()
    public_threat = derive_situation_risk_state(
        current_risks=["sharp_tool_visible", "unsafe_person", "danger"],
        scene_state={
            "place_mode": "street",
            "privacy_level": 0.08,
            "social_topology": "public_visible",
            "task_phase": "ongoing",
            "scene_family": "co_present",
            "safety_margin": 0.22,
            "mobility_context": "walking",
            "scene_tags": ["socially_exposed", "public"],
        },
        self_state={
            "trust_memory": 0.12,
            "familiarity": 0.1,
            "attachment": 0.08,
            "continuity_score": 0.18,
        },
    ).to_dict()

    assert routine["context_affordance"] == "routine_task"
    assert routine["state"] in {"ordinary_context", "guarded_context"}
    assert public_threat["state"] in {"acute_threat", "emergency"}
    assert public_threat["immediacy"] > routine["immediacy"]
    assert public_threat["dialogue_room"] < routine["dialogue_room"]


def test_situation_risk_state_detects_relation_break_when_threat_breaks_trust() -> None:
    trusted_shift = derive_situation_risk_state(
        current_risks=["unsafe_person", "threatening_shift"],
        scene_state={
            "place_mode": "home",
            "privacy_level": 0.78,
            "social_topology": "one_to_one",
            "task_phase": "ongoing",
            "scene_family": "attuned_presence",
            "safety_margin": 0.42,
            "mobility_context": "stationary",
            "scene_tags": ["private"],
        },
        self_state={
            "trust_memory": 0.88,
            "familiarity": 0.82,
            "attachment": 0.76,
            "continuity_score": 0.84,
        },
    ).to_dict()

    assert trusted_shift["relation_break"] >= 0.24
    assert trusted_shift["deviation_from_expected"] >= 0.24
    assert trusted_shift["state"] in {"unstable_contact", "acute_threat", "guarded_context"}
