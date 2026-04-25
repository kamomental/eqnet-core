from inner_os.orchestration.stimulus_history_influence import (
    apply_stimulus_history_influence_to_contract_inputs,
    derive_stimulus_history_influence,
)


def test_stimulus_history_influence_holds_when_spike_is_foggy() -> None:
    influence = derive_stimulus_history_influence(
        {
            "qualia_structure_state": {
                "emergence": 0.82,
                "drift": 0.76,
                "intensity": 0.7,
                "stability": 0.16,
            },
            "qualia_state": {
                "habituation": [0.05, 0.08, 0.02],
                "normalization_stats": {
                    "value_grad": {
                        "range_trust": 0.18,
                        "gradient_confidence": 0.12,
                    }
                },
            },
            "environment": {"fog_density": 0.72},
        }
    )

    projected = apply_stimulus_history_influence_to_contract_inputs(
        contract_inputs=_base_contract_inputs(),
        influence=influence,
    )

    assert influence.response_bias == "hold_for_clarity"
    assert projected["actuation_plan"]["response_channel"] == "hold"
    assert projected["discourse_shape"]["shape_id"] == "reflect_hold"
    assert projected["surface_context_packet"]["source_state"]["stimulus_field_clarity"] < 0.5


def test_stimulus_history_influence_keeps_habituated_clear_field_speakable() -> None:
    influence = derive_stimulus_history_influence(
        {
            "qualia_structure_state": {
                "emergence": 0.28,
                "drift": 0.08,
                "intensity": 0.24,
                "stability": 0.78,
                "memory_resonance": 0.42,
            },
            "qualia_state": {
                "habituation": [0.72, 0.68, 0.7],
                "normalization_stats": {
                    "value_grad": {
                        "range_trust": 0.84,
                        "gradient_confidence": 0.8,
                    },
                    "habituation": {
                        "range_trust": 0.78,
                        "gradient_confidence": 0.74,
                    },
                },
            },
        }
    )
    projected = apply_stimulus_history_influence_to_contract_inputs(
        contract_inputs=_base_contract_inputs(),
        influence=influence,
    )

    assert influence.response_bias == "habituated_small_response"
    assert projected["actuation_plan"]["response_channel"] == "speak"
    assert (
        projected["surface_context_packet"]["source_state"][
            "stimulus_habituated_response"
        ]
        == influence.habituation_pressure
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
