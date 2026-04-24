from inner_os.evaluation.conversation_contract_eval import (
    CORE_QUICKSTART_EXPECTATIONS,
    evaluate_reaction_contract_against_expectation,
)
from inner_os.evaluation.prompt_baseline_fixtures import prompt_baselines_for_scenario


def test_small_shared_moment_prompt_baselines_expose_contract_violations() -> None:
    baselines = prompt_baselines_for_scenario("small_shared_moment")

    assert len(baselines) >= 2
    failed = [
        evaluate_reaction_contract_against_expectation(
            reaction_contract=sample.observed_contract,
            expectation=CORE_QUICKSTART_EXPECTATIONS["small_shared_moment"],
        )
        for sample in baselines
    ]
    assert all(result.passed is False for result in failed)


def test_guarded_uncertainty_prompt_baseline_exposes_hold_violation() -> None:
    baselines = prompt_baselines_for_scenario("guarded_uncertainty")

    assert len(baselines) >= 1
    result = evaluate_reaction_contract_against_expectation(
        reaction_contract=baselines[0].observed_contract,
        expectation=CORE_QUICKSTART_EXPECTATIONS["guarded_uncertainty"],
    )
    codes = {violation.code for violation in result.violations}
    assert result.passed is False
    assert "response_channel_mismatch" in codes
    assert "question_budget_violation" in codes
