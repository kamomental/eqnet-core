from scripts.core_quickstart_demo import build_core_demo_result


def test_small_shared_moment_demo_stays_small_and_joined() -> None:
    result = build_core_demo_result(scenario_name="small_shared_moment")

    contract = result["reaction_contract"]
    assert contract["stance"] == "join"
    assert contract["scale"] == "small"
    assert contract["question_budget"] == 0
    assert contract["interpretation_budget"] == "none"
    assert result["evaluation"]["passed"] is True
    assert result["expected_contract"]["stance"] == "join"


def test_guarded_uncertainty_demo_prefers_hold() -> None:
    result = build_core_demo_result(scenario_name="guarded_uncertainty")

    contract = result["reaction_contract"]
    assert contract["stance"] == "hold"
    assert contract["response_channel"] == "hold"
    assert contract["continuity_mode"] == "reopen"
    assert result["evaluation"]["passed"] is True
    assert result["expected_contract"]["timing_mode"] == "held_open"
