from scripts.core_quickstart_demo import build_core_demo_result


def test_core_quickstart_reaction_changes_with_stimulus_history_context() -> None:
    text = "today feels a little tired"
    calm = build_core_demo_result(
        scenario_name="small_shared_moment",
        input_text=text,
        expression_context_state={
            "qualia_structure_state": {
                "emergence": 0.24,
                "drift": 0.08,
                "intensity": 0.22,
                "stability": 0.8,
            },
            "qualia_state": {
                "habituation": [0.68, 0.7, 0.66],
                "normalization_stats": {
                    "value_grad": {
                        "range_trust": 0.82,
                        "gradient_confidence": 0.78,
                    }
                },
            },
        },
    )
    foggy_spike = build_core_demo_result(
        scenario_name="small_shared_moment",
        input_text=text,
        expression_context_state={
            "qualia_structure_state": {
                "emergence": 0.86,
                "drift": 0.78,
                "intensity": 0.74,
                "stability": 0.18,
            },
            "qualia_state": {
                "habituation": [0.03, 0.06, 0.02],
                "normalization_stats": {
                    "value_grad": {
                        "range_trust": 0.16,
                        "gradient_confidence": 0.12,
                    }
                },
            },
            "environment": {"fog_density": 0.75},
        },
    )

    assert calm["stimulus_history_influence"]["response_bias"] == "habituated_small_response"
    assert foggy_spike["stimulus_history_influence"]["response_bias"] == "hold_for_clarity"
    assert calm["reaction_contract"]["response_channel"] == "speak"
    assert foggy_spike["reaction_contract"]["response_channel"] == "hold"
    assert calm["llm_expression_request"]["should_call_llm"] is True
    assert foggy_spike["llm_expression_request"]["should_call_llm"] is False


def test_core_quickstart_reaction_changes_with_protective_trace_context() -> None:
    text = "today feels a little tired"
    ordinary = build_core_demo_result(
        scenario_name="small_shared_moment",
        input_text=text,
        expression_context_state={
            "qualia_structure_state": {
                "emergence": 0.24,
                "drift": 0.08,
                "intensity": 0.22,
                "stability": 0.8,
            },
            "qualia_state": {
                "habituation": [0.68, 0.7, 0.66],
                "normalization_stats": {
                    "value_grad": {
                        "range_trust": 0.82,
                        "gradient_confidence": 0.78,
                    }
                },
            },
        },
    )
    protective = build_core_demo_result(
        scenario_name="small_shared_moment",
        input_text=text,
        expression_context_state={
            "memory": {
                "memory_write_class": "body_risk",
                "ignition_readiness": 0.72,
                "memory_tension": 0.68,
                "present_threat_binding": 0.72,
            },
            "body": {
                "stress": 0.76,
                "somatic_reactivation": 0.82,
                "hyperarousal": 0.76,
            },
            "protective_trace": {
                "sensory_flash_risk": 0.78,
                "dream_intrusion_pressure": 0.34,
                "trigger_pressure": 0.68,
            },
            "qualia_structure_state": {
                "emergence": 0.74,
                "drift": 0.7,
                "intensity": 0.68,
                "stability": 0.2,
            },
            "qualia_state": {
                "habituation": [0.04, 0.06, 0.03],
                "normalization_stats": {
                    "value_grad": {
                        "range_trust": 0.18,
                        "gradient_confidence": 0.14,
                    }
                },
            },
        },
    )

    assert ordinary["protective_trace_palace"]["dominant_mode"] in {
        "ambient",
        "recovery_opening",
    }
    assert protective["protective_trace_palace"]["dominant_mode"] == "protective_hold"
    assert ordinary["reaction_contract"]["response_channel"] == "speak"
    assert protective["reaction_contract"]["response_channel"] == "hold"
    assert ordinary["llm_expression_request"]["should_call_llm"] is True
    assert protective["llm_expression_request"]["should_call_llm"] is False
    assert (
        protective["quick_audit_projection"]["audit_axes"][
            "protective_trace_dominant_mode"
        ]
        == "protective_hold"
    )


def test_core_quickstart_exposes_current_crisis_and_rem_replay_axes() -> None:
    result = build_core_demo_result(
        scenario_name="small_shared_moment",
        input_text="today feels a little tired",
        expression_context_state={
            "memory": {
                "memory_write_class": "body_risk",
                "present_threat_binding": 0.86,
                "trigger_match": 0.72,
            },
            "body": {
                "hyperarousal": 0.8,
                "startle": 0.7,
            },
            "environment": {
                "trigger_salience": 0.74,
            },
            "sleep": {
                "rem_replay_pressure": 0.68,
            },
            "qualia_structure_state": {
                "emergence": 0.6,
                "drift": 0.52,
                "intensity": 0.58,
                "stability": 0.24,
            },
            "qualia_state": {
                "habituation": [0.12, 0.08, 0.1],
                "normalization_stats": {
                    "value_grad": {
                        "range_trust": 0.26,
                        "gradient_confidence": 0.2,
                    }
                },
            },
        },
    )
    axes = result["quick_audit_projection"]["audit_axes"]

    assert result["protective_trace_palace"]["dominant_mode"] == "protective_hold"
    assert axes["protective_trace_current_crisis_binding"] >= 0.55
    assert axes["protective_trace_hyperarousal_pressure"] >= 0.45
    assert axes["protective_trace_trigger_pressure"] >= 0.45
    assert axes["protective_trace_rem_replay_pressure"] >= 0.55
    assert result["llm_expression_request"]["should_call_llm"] is False
