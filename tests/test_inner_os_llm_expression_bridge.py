from inner_os.expression.llm_expression_bridge import build_llm_expression_request


def test_llm_expression_request_uses_state_contract_for_speak() -> None:
    request = build_llm_expression_request(
        input_text="さっきの続きなんだけど、あのあとちょっと笑えることもあって。",
        reaction_contract={
            "stance": "join",
            "scale": "small",
            "response_channel": "speak",
            "question_budget": 0,
            "interpretation_budget": "none",
            "continuity_mode": "continue",
        },
        joint_state={"common_ground": 0.74, "shared_delight": 0.48},
        shared_presence={"dominant_mode": "inhabited_shared_space"},
    )

    assert request.should_call_llm is True
    assert request.action_channel == "speak"
    assert request.contract["stance"] == "join"
    assert "質問で終えない" in request.user_prompt
    assert "断定解釈しない" in request.user_prompt
    assert "shared_delight" in request.user_prompt
    assert request.fallback_action["type"] == "review_gate"


def test_llm_expression_request_skips_llm_for_hold_action() -> None:
    request = build_llm_expression_request(
        input_text="まだうまく言葉にできないんだけど、ちょっと重い感じだけ残ってて。",
        reaction_contract={
            "stance": "hold",
            "response_channel": "hold",
            "question_budget": 0,
            "timing_mode": "held_open",
        },
        joint_state={"shared_tension": 0.62},
    )

    assert request.should_call_llm is False
    assert request.action_channel == "hold"
    assert request.system_prompt == ""
    assert request.user_prompt == ""
    assert request.blocked_reason
    assert request.fallback_action["type"] == "nonverbal"
