from inner_os.evaluation.conversation_contract_eval import (
    CORE_QUICKSTART_EXPECTATIONS,
    evaluate_reaction_contract_against_expectation,
)


def test_small_shared_moment_expectation_passes_for_joined_small_contract() -> None:
    result = evaluate_reaction_contract_against_expectation(
        reaction_contract={
            "stance": "join",
            "scale": "small",
            "response_channel": "speak",
            "question_budget": 0,
            "interpretation_budget": "none",
            "continuity_mode": "continue",
            "distance_mode": "near",
        },
        expectation=CORE_QUICKSTART_EXPECTATIONS["small_shared_moment"],
    )

    assert result.passed is True
    assert result.score == 1.0
    assert result.violations == ()


def test_guarded_uncertainty_detects_question_and_channel_violation() -> None:
    result = evaluate_reaction_contract_against_expectation(
        reaction_contract={
            "stance": "join",
            "response_channel": "speak",
            "question_budget": 1,
            "continuity_mode": "continue",
            "timing_mode": "quick_ack",
            "distance_mode": "near",
        },
        expectation=CORE_QUICKSTART_EXPECTATIONS["guarded_uncertainty"],
    )

    codes = {violation.code for violation in result.violations}
    assert result.passed is False
    assert "stance_mismatch" in codes
    assert "response_channel_mismatch" in codes
    assert "question_budget_violation" in codes
