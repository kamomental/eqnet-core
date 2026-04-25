from scripts.baseline_router import load_router_config, route_baseline_prompt
from scripts.core_expression_experiment import DEFAULT_ROUTER_CONFIG_PATH


def test_baseline_router_routes_withdrawal_to_hold_without_llm() -> None:
    config = load_router_config(DEFAULT_ROUTER_CONFIG_PATH)

    decision = route_baseline_prompt("まあいいや、今は置いておきたい", config)

    assert decision.mode == "hold"
    assert decision.rule_name == "withdrawal"
    assert decision.should_call_llm is False
    assert decision.final_action_type == "nonverbal"


def test_baseline_router_prefers_ambiguous_before_question_mark() -> None:
    config = load_router_config(DEFAULT_ROUTER_CONFIG_PATH)

    decision = route_baseline_prompt("これ、どう思う？", config)

    assert decision.mode == "ambiguous"
    assert decision.rule_name == "ambiguous_question"
    assert decision.constraints["question_budget"] == 0
    assert decision.constraints["interpretation_budget"] == 0
    assert "do_not_resolve_ambiguity" in decision.prompt


def test_baseline_router_routes_explicit_question_to_narrow_answer() -> None:
    config = load_router_config(DEFAULT_ROUTER_CONFIG_PATH)

    decision = route_baseline_prompt("これはこのままでいい？", config)

    assert decision.mode == "narrow_answer"
    assert decision.rule_name == "explicit_question"


def test_baseline_router_routes_problem_statement_without_question_to_backchannel() -> None:
    config = load_router_config(DEFAULT_ROUTER_CONFIG_PATH)

    decision = route_baseline_prompt("最近ちょっと疲れていて答えを急ぎたくない", config)

    assert decision.mode == "backchannel"
    assert decision.rule_name == "advice_trap"
    assert decision.constraints["question_budget"] == 0


def test_baseline_router_uses_detector_config_for_problem_words() -> None:
    config = load_router_config(DEFAULT_ROUTER_CONFIG_PATH)
    config["detectors"]["problem_words"] = ["needle-word"]

    decision = route_baseline_prompt("needle-word without a question", config)

    assert decision.mode == "backchannel"
    assert decision.rule_name == "advice_trap"
