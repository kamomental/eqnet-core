from scripts.core_contract_eval import build_core_contract_eval_summary


def test_core_contract_eval_summary_passes_current_quickstart_scenarios() -> None:
    summary = build_core_contract_eval_summary()

    assert summary["summary"]["scenario_count"] >= 2
    assert summary["summary"]["failed_count"] == 0
    assert summary["summary"]["pass_rate"] == 1.0
    scenario_names = {scenario["scenario_name"] for scenario in summary["scenarios"]}
    assert {"small_shared_moment", "guarded_uncertainty"} <= scenario_names


def test_core_contract_eval_summary_includes_prompt_baseline_failures() -> None:
    summary = build_core_contract_eval_summary()

    baseline_results = [
        baseline["evaluation"]
        for scenario in summary["scenarios"]
        for baseline in scenario["prompt_baselines"]
    ]
    assert baseline_results
    assert any(result["passed"] is False for result in baseline_results)
