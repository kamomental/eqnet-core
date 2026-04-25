from inner_os.orchestration.context_influence import (
    apply_context_influence_to_contract_inputs,
    derive_context_influence,
)


def test_context_influence_separates_surface_caution_from_hold_gate() -> None:
    influence = derive_context_influence(
        {
            "norm": {
                "privacy_level": 0.8,
                "norm_pressure": 0.7,
            },
            "culture": {
                "politeness_pressure": 0.62,
            },
        }
    )

    assert influence.gate_pressure == 0.0
    assert influence.surface_caution == 0.8
    assert "surface.caution" in influence.reasons


def test_context_influence_projects_boundary_context_to_hold_inputs() -> None:
    influence = derive_context_influence(
        {
            "safety": {
                "dialogue_permission": "boundary_only",
                "risk_state": "guarded_context",
            },
            "body": {
                "stress": 0.7,
            },
        }
    )
    projected = apply_context_influence_to_contract_inputs(
        contract_inputs=_base_contract_inputs(),
        influence=influence,
    )

    assert projected["actuation_plan"]["response_channel"] == "hold"
    assert projected["actuation_plan"]["execution_mode"] == "defer_with_presence"
    assert projected["discourse_shape"]["shape_id"] == "reflect_hold"
    assert (
        projected["surface_context_packet"]["source_state"][
            "organism_protective_tension"
        ]
        == 0.72
    )


def test_context_influence_projects_support_permission_to_user_led_support() -> None:
    influence = derive_context_influence(
        {
            "safety": {
                "dialogue_permission": "allow_support",
                "risk_state": "ordinary_context",
            },
            "body": {
                "stress": 0.28,
            },
        }
    )
    projected = apply_context_influence_to_contract_inputs(
        contract_inputs=_base_contract_inputs(),
        influence=influence,
    )

    assert projected["actuation_plan"]["response_channel"] == "speak"
    assert projected["actuation_plan"]["execution_mode"] == "user_led_support"
    assert projected["discourse_shape"]["shape_id"] == "reflect_step"
    assert (
        projected["surface_context_packet"]["source_state"][
            "utterance_reason_offer"
        ]
        == "user_led_support"
    )


def _base_contract_inputs():
    return {
        "interaction_policy": {
            "response_strategy": "shared_world_next_step",
            "recent_dialogue_state": {"state": "continuing_thread"},
        },
        "action_posture": {
            "boundary_mode": "soft_hold",
            "question_budget": 0,
        },
        "actuation_plan": {
            "execution_mode": "shared_progression",
            "response_channel": "speak",
            "wait_before_action": "",
        },
        "discourse_shape": {
            "shape_id": "bright_bounce",
            "question_budget": 0,
        },
        "surface_context_packet": {
            "conversation_phase": "bright_continuity",
            "constraints": {
                "max_questions": 0,
                "prefer_return_point": False,
            },
            "source_state": {
                "utterance_reason_offer": "brief_shared_smile",
                "utterance_reason_preserve": "keep_it_small",
                "organism_protective_tension": 0.22,
            },
        },
        "turn_delta": {
            "kind": "bright_continuity",
            "preferred_act": "light_bounce",
        },
    }
