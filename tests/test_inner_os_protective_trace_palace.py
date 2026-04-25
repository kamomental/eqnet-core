from inner_os.orchestration.protective_trace_palace import (
    apply_protective_trace_palace_to_contract_inputs,
    derive_protective_trace_palace_state,
)
from inner_os.orchestration.stimulus_history_influence import StimulusHistoryInfluence


def test_protective_trace_holds_when_reentry_is_sensitive_and_unclear() -> None:
    state = derive_protective_trace_palace_state(
        {
            "memory": {
                "memory_write_class": "body_risk",
                "ignition_readiness": 0.72,
                "memory_tension": 0.68,
            },
            "body": {
                "stress": 0.76,
                "somatic_reactivation": 0.82,
            },
            "protective_trace": {
                "sensory_flash_risk": 0.78,
                "dream_intrusion_pressure": 0.34,
            },
            "safety": {"dialogue_permission": "boundary_only"},
        },
        stimulus_history_influence=StimulusHistoryInfluence(
            stimulus_pressure=0.72,
            novelty_pressure=0.68,
            memory_reentry_pressure=0.7,
            field_clarity=0.18,
            gradient_confidence=0.16,
        ),
    )

    projected = apply_protective_trace_palace_to_contract_inputs(
        contract_inputs=_base_contract_inputs(),
        palace_state=state,
    )

    assert state.dominant_mode == "protective_hold"
    assert state.stabilization_need >= 0.58
    assert projected["actuation_plan"]["response_channel"] == "hold"
    assert projected["discourse_shape"]["shape_id"] == "reflect_hold"
    assert (
        projected["surface_context_packet"]["source_state"][
            "utterance_reason_preserve"
        ]
        == "stabilize_before_reentry"
    )


def test_protective_trace_allows_small_recovery_opening_when_stable() -> None:
    state = derive_protective_trace_palace_state(
        {
            "memory": {
                "memory_write_class": "repair_trace",
                "repair_trace": 0.72,
                "safe_repeat": 0.68,
                "memory_tension": 0.18,
            },
            "homeostasis": {
                "recovery_capacity": 0.78,
                "load": 0.12,
            },
            "protective_trace": {
                "trusted_reentry_window": 0.8,
                "recovery_path_strength": 0.76,
                "sensory_flash_risk": 0.08,
            },
        },
        stimulus_history_influence=StimulusHistoryInfluence(
            stimulus_pressure=0.18,
            novelty_pressure=0.12,
            memory_reentry_pressure=0.2,
            field_clarity=0.84,
            gradient_confidence=0.8,
        ),
    )

    projected = apply_protective_trace_palace_to_contract_inputs(
        contract_inputs=_base_contract_inputs(),
        palace_state=state,
    )

    assert state.dominant_mode == "safe_reconsolidation"
    assert projected["actuation_plan"]["response_channel"] == "speak"
    assert projected["discourse_shape"]["shape_id"] == "reflect_step"
    assert (
        projected["surface_context_packet"]["source_state"][
            "utterance_reason_preserve"
        ]
        == "safe_reconsolidation_window"
    )


def test_current_crisis_binding_and_hyperarousal_force_protective_hold() -> None:
    state = derive_protective_trace_palace_state(
        {
            "memory": {
                "memory_write_class": "body_risk",
                "present_threat_binding": 0.88,
                "trigger_match": 0.74,
            },
            "body": {
                "hyperarousal": 0.82,
                "startle": 0.76,
            },
            "environment": {
                "trigger_salience": 0.78,
            },
            "protective_trace": {
                "current_crisis_binding": 0.9,
            },
        },
        stimulus_history_influence=StimulusHistoryInfluence(
            stimulus_pressure=0.54,
            novelty_pressure=0.48,
            memory_reentry_pressure=0.44,
            field_clarity=0.3,
            gradient_confidence=0.24,
        ),
    )
    projected = apply_protective_trace_palace_to_contract_inputs(
        contract_inputs=_base_contract_inputs(),
        palace_state=state,
    )

    assert state.current_crisis_binding >= 0.55
    assert state.hyperarousal_pressure >= 0.45
    assert state.trigger_pressure >= 0.45
    assert state.dominant_mode == "protective_hold"
    assert "protective_trace.current_crisis_binding" in state.reasons
    assert projected["actuation_plan"]["response_channel"] == "hold"


def test_rem_replay_and_dream_intrusion_choose_restabilize() -> None:
    state = derive_protective_trace_palace_state(
        {
            "memory": {
                "memory_write_class": "repair_trace",
                "memory_tension": 0.28,
            },
            "sleep": {
                "rem_replay_pressure": 0.82,
                "dream_intrusion_pressure": 0.74,
                "nightmare_pressure": 0.68,
            },
            "body": {
                "stress": 0.2,
            },
        },
        stimulus_history_influence=StimulusHistoryInfluence(
            stimulus_pressure=0.22,
            novelty_pressure=0.18,
            memory_reentry_pressure=0.24,
            field_clarity=0.72,
            gradient_confidence=0.68,
        ),
    )

    assert state.rem_replay_pressure >= 0.55
    assert state.dream_intrusion_pressure >= 0.55
    assert state.dominant_mode == "restabilize"
    assert "sleep.rem_replay" in state.reasons


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
