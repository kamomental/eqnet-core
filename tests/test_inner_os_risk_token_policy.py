from dataclasses import asdict

from inner_os.emergency_posture import derive_emergency_posture
from inner_os.risk_token_policy import derive_current_risk_token_policy
from inner_os.scene_state import derive_scene_state
from inner_os.situation_risk_state import derive_situation_risk_state


def test_risk_token_policy_keeps_bright_public_followup_out_of_danger() -> None:
    policy = derive_current_risk_token_policy(
        safety_bias=0.36,
        stress=0.14,
        recovery_need=0.08,
        recent_strain=0.08,
        privacy_level=0.14,
        social_topology="threaded_group",
        norm_pressure=0.26,
        safety_margin=0.64,
        environmental_load=0.16,
        mobility_context="stationary",
        task_phase="ongoing",
    )
    scene = derive_scene_state(
        place_mode="stream",
        privacy_level=0.14,
        social_topology="threaded_group",
        task_phase="ongoing",
        temporal_phase="ongoing",
        norm_pressure=0.26,
        safety_margin=0.64,
        environmental_load=0.16,
        mobility_context="stationary",
        current_risks=list(policy.tokens),
        active_goals=[],
    )
    risk = derive_situation_risk_state(
        current_risks=list(policy.tokens),
        scene_state=asdict(scene),
        self_state={
            "trust_memory": 0.74,
            "familiarity": 0.68,
            "attachment": 0.66,
            "continuity_score": 0.68,
        },
    ).to_dict()
    posture = derive_emergency_posture(
        situation_risk_state=risk,
        constraint_field={
            "boundary_pressure": 0.34,
            "protective_bias": 0.36,
        },
        protection_mode={"mode": "contain", "strength": 0.34},
        body_recovery_guard={"state": "open", "score": 0.08},
        body_homeostasis_state={"state": "steady", "score": 0.1},
        homeostasis_budget_state={"state": "steady", "score": 0.1},
    ).to_dict()

    assert "danger" not in policy.tokens
    assert scene.scene_family != "guarded_boundary"
    assert risk["state"] in {"ordinary_context", "guarded_context"}
    assert posture["state"] in {"observe", "de_escalate"}
    assert posture["dialogue_permission"] != "avoid_dialogue"


def test_risk_token_policy_emits_danger_when_pressure_is_acute() -> None:
    policy = derive_current_risk_token_policy(
        safety_bias=0.72,
        stress=0.62,
        recovery_need=0.58,
        recent_strain=0.66,
        privacy_level=0.08,
        social_topology="public_visible",
        norm_pressure=0.64,
        safety_margin=0.18,
        environmental_load=0.42,
        mobility_context="walking",
        task_phase="ongoing",
    )

    assert "danger" in policy.tokens
    assert policy.acute_pressure >= 0.42
