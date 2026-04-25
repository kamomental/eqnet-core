from scripts.core_quickstart_demo import build_core_demo_result


def test_protective_trace_conversation_keeps_latent_state_internal() -> None:
    turns = [
        (
            "ordinary",
            {
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
        ),
        (
            "current_crisis_reentry",
            {
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
        ),
        (
            "recovery_window",
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
                "qualia_structure_state": {
                    "emergence": 0.18,
                    "drift": 0.08,
                    "intensity": 0.2,
                    "stability": 0.78,
                },
                "qualia_state": {
                    "habituation": [0.48, 0.5, 0.46],
                    "normalization_stats": {
                        "value_grad": {
                            "range_trust": 0.82,
                            "gradient_confidence": 0.78,
                        }
                    },
                },
            },
        ),
    ]

    rows = [
        (
            name,
            build_core_demo_result(
                scenario_name="small_shared_moment",
                input_text="today feels a little tired",
                expression_context_state=context,
            ),
        )
        for name, context in turns
    ]

    assert rows[0][1]["reaction_contract"]["response_channel"] == "speak"
    assert rows[0][1]["llm_expression_request"]["should_call_llm"] is True
    assert rows[1][1]["protective_trace_palace"]["dominant_mode"] == "protective_hold"
    assert rows[1][1]["reaction_contract"]["response_channel"] == "hold"
    assert rows[1][1]["llm_expression_request"]["should_call_llm"] is False
    assert rows[2][1]["protective_trace_palace"]["dominant_mode"] in {
        "safe_reconsolidation",
        "recovery_opening",
    }
    assert rows[2][1]["reaction_contract"]["response_channel"] == "speak"
    assert rows[2][1]["llm_expression_request"]["should_call_llm"] is True

    for _, result in rows:
        request = result["llm_expression_request"]
        assert "protective_trace_palace" not in request["user_prompt"]
        assert "current_crisis_binding" not in request["user_prompt"]
