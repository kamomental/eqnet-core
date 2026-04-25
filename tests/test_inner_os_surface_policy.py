from inner_os.expression.surface_policy import compile_surface_policy


def test_compile_surface_policy_turns_hold_contract_into_nonverbal_policy() -> None:
    policy = compile_surface_policy(
        {
            "response_channel": "hold",
            "scale": "micro",
            "question_budget": 0,
            "interpretation_budget": "low",
            "initiative": "yield",
            "distance_mode": "guarded",
            "timing_mode": "held_open",
            "closure_mode": "leave_open",
        }
    )

    assert policy.response_channel == "hold"
    assert policy.max_sentences == 0
    assert policy.question_budget == 0
    assert policy.advice_budget == 0
    assert policy.brightness_budget == 0
    assert "nonverbal_presence" in policy.allowed_acts
    assert "generate_text" in policy.prohibited_acts
    assert "ask_question" in policy.prohibited_acts
    assert policy.fallback_shape_id == "presence_hold"


def test_compile_surface_policy_keeps_speak_contract_as_surface_view() -> None:
    policy = compile_surface_policy(
        {
            "response_channel": "speak",
            "scale": "small",
            "question_budget": 0,
            "interpretation_budget": "none",
            "initiative": "receive",
            "distance_mode": "steady",
            "timing_mode": "quick_ack",
            "shape_id": "bright_bounce",
        }
    )

    assert policy.response_channel == "speak"
    assert policy.max_sentences == 2
    assert policy.question_budget == 0
    assert policy.interpretation_budget == "none"
    assert policy.advice_budget == 0
    assert policy.brightness_budget == 0
    assert "natural_surface_text" in policy.allowed_acts
    assert "ask_question" in policy.prohibited_acts
    assert "infer_hidden_feeling" in policy.prohibited_acts
    assert policy.fallback_shape_id == "bright_bounce_minimal"
