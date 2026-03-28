from types import SimpleNamespace

from inner_os.action_posture import derive_action_posture
from inner_os.actuation_plan import derive_actuation_plan
from inner_os.policy_packet import derive_interaction_policy_packet


def test_policy_packet_emergency_posture_biases_posture_and_actuation() -> None:
    packet = derive_interaction_policy_packet(
        dialogue_act="check_in",
        current_focus="person:user",
        current_risks=["sharp_tool_visible", "unsafe_person", "danger"],
        reportable_facts=["there is visible risk nearby"],
        relation_bias_strength=0.2,
        related_person_ids=["person:user"],
        partner_address_hint="unknown",
        partner_timing_hint="delayed",
        partner_stance_hint="uncertain",
        orchestration={
            "orchestration_mode": "contain",
            "dominant_driver": "safety",
            "contact_readiness": 0.22,
            "coherence_score": 0.3,
            "human_presence_signal": 0.34,
        },
        surface_profile={
            "opening_pace_windowed": "measured",
            "return_gaze_expectation": "soft_return",
        },
        live_regulation=SimpleNamespace(
            distance_expectation="respectful_distance",
            repair_window_open=False,
            strained_pause=0.1,
            future_loop_pull=0.04,
            fantasy_loop_pull=0.0,
        ),
        scene_state={
            "place_mode": "street",
            "privacy_level": 0.08,
            "social_topology": "public_visible",
            "task_phase": "ongoing",
            "temporal_phase": "ongoing",
            "norm_pressure": 0.62,
            "safety_margin": 0.22,
            "environmental_load": 0.28,
            "mobility_context": "walking",
            "scene_family": "guarded_boundary",
            "scene_tags": ["socially_exposed", "public"],
        },
        conscious_workspace={
            "workspace_mode": "guarded_foreground",
            "workspace_stability": 0.48,
            "reportable_slice": ["there is visible risk nearby"],
            "actionable_slice": ["increase distance first"],
            "reportability_gate": {"gate_mode": "narrow"},
        },
        affect_blend_state={
            "care": 0.22,
            "defense": 0.58,
            "reverence": 0.08,
            "conflict_level": 0.34,
            "residual_tension": 0.4,
            "shared_world_pull": 0.08,
            "future_pull": 0.06,
            "distress": 0.44,
        },
        constraint_field={
            "body_cost": 0.24,
            "boundary_pressure": 0.68,
            "repair_pressure": 0.12,
            "shared_world_pressure": 0.08,
            "protective_bias": 0.64,
            "disclosure_limit": "minimal",
            "reportability_limit": "withhold",
            "option_temperature": 0.9,
            "admissible_families": ["wait", "contain", "withdraw"],
            "do_not_cross": ["force_reportability"],
            "cues": ["constraint_withhold"],
        },
        resonance_evaluation={
            "estimated_other_person_state": {
                "detail_room_level": "low",
                "acknowledgement_need_level": "low",
                "pressure_sensitivity_level": "high",
                "next_step_room_level": "low",
            },
        },
        qualia_planner_view={
            "trust": 0.22,
            "degraded": False,
            "dominant_axis": "protection",
            "dominant_value": 0.26,
            "body_load": 0.12,
            "protection_bias": 0.2,
            "felt_energy": 0.1,
        },
        terrain_readout={
            "value": -0.12,
            "grad": [0.0, 0.0, 0.0, 0.0],
            "curvature": [0.0, 0.0, 0.0, 0.0],
            "approach_bias": 0.08,
            "avoid_bias": 0.42,
            "protect_bias": 0.36,
            "active_patch_index": 0,
            "active_patch_label": "risk_field",
        },
        protection_mode={
            "mode": "monitor",
            "strength": 0.32,
            "reasons": ["visible_risk"],
        },
        self_state={
            "stress": 0.22,
            "recovery_need": 0.16,
            "recent_strain": 0.14,
            "safety_bias": 0.26,
            "continuity_score": 0.22,
            "social_grounding": 0.18,
            "trust_memory": 0.12,
            "familiarity": 0.08,
            "attachment": 0.08,
        },
    )

    posture = derive_action_posture(packet)
    actuation = derive_actuation_plan(packet, posture)

    assert packet["situation_risk_state"]["state"] in {"acute_threat", "emergency"}
    assert packet["emergency_posture"]["state"] in {"create_distance", "exit", "seek_help", "emergency_protect"}
    assert posture["emergency_posture_name"] == packet["emergency_posture"]["state"]
    assert posture["situation_risk_name"] == packet["situation_risk_state"]["state"]
    assert any(
        item in posture["next_action_candidates"]
        for item in {"create_distance", "exit_space", "seek_help", "emergency_protect"}
    )
    assert actuation["emergency_posture_name"] == packet["emergency_posture"]["state"]
    assert actuation["primary_action"] in {"create_distance", "exit_space", "seek_help", "protect_immediately"}
    assert any(
        item in actuation["action_queue"]
        for item in {"orient_to_exit", "move_to_safety", "make_risk_visible", "protect_others_if_present"}
    )
