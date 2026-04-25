from scripts.core_quickstart_demo import build_core_demo_result


def test_core_quickstart_reaction_changes_with_stimulus_history_context() -> None:
    text = "今日は少し疲れた。"
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
