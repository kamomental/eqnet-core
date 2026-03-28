from inner_os.emergency_posture import derive_emergency_posture
from inner_os.situation_risk_state import derive_situation_risk_state


def test_emergency_posture_prefers_create_distance_for_public_acute_threat() -> None:
    risk = derive_situation_risk_state(
        current_risks=["sharp_tool_visible", "unsafe_person", "danger"],
        scene_state={
            "place_mode": "street",
            "privacy_level": 0.08,
            "social_topology": "public_visible",
            "task_phase": "ongoing",
            "scene_family": "co_present",
            "safety_margin": 0.28,
            "mobility_context": "walking",
            "scene_tags": ["socially_exposed", "public"],
        },
        self_state={},
    ).to_dict()
    posture = derive_emergency_posture(
        situation_risk_state=risk,
        constraint_field={"boundary_pressure": 0.52, "protective_bias": 0.48},
        protection_mode={"mode": "monitor", "strength": 0.26},
        body_recovery_guard={"state": "open", "score": 0.08},
        body_homeostasis_state={"state": "steady", "score": 0.1},
        homeostasis_budget_state={"state": "steady", "score": 0.08},
    ).to_dict()

    assert posture["state"] == "create_distance"
    assert posture["dialogue_permission"] == "boundary_only"
    assert posture["primary_action"] == "create_distance"


def test_emergency_posture_prefers_seek_help_or_exit_for_private_intrusion() -> None:
    risk = derive_situation_risk_state(
        current_risks=["forced_entry", "intrusion", "danger"],
        scene_state={
            "place_mode": "home",
            "privacy_level": 0.86,
            "social_topology": "one_to_one",
            "task_phase": "ongoing",
            "scene_family": "co_present",
            "safety_margin": 0.16,
            "mobility_context": "stationary",
            "scene_tags": ["private"],
        },
        self_state={},
    ).to_dict()
    posture = derive_emergency_posture(
        situation_risk_state=risk,
        constraint_field={"boundary_pressure": 0.72, "protective_bias": 0.7},
        protection_mode={"mode": "shield", "strength": 0.82},
        body_recovery_guard={"state": "guarded", "score": 0.48},
        body_homeostasis_state={"state": "steady", "score": 0.12},
        homeostasis_budget_state={"state": "steady", "score": 0.12},
    ).to_dict()

    assert posture["state"] in {"seek_help", "exit", "emergency_protect"}
    assert posture["dialogue_permission"] == "avoid_dialogue"
    assert posture["primary_action"] in {"seek_help", "exit_space", "protect_immediately"}
