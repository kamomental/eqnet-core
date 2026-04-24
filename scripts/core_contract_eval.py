from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.core_quickstart_demo import (  # noqa: E402
    CORE_QUICKSTART_EXPECTATIONS,
    SCENARIOS,
    build_core_demo_result,
    evaluate_reaction_contract_against_expectation,
)


def _load_module(module_name: str, relative_path: str) -> Any:
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_BASELINE_MODULE = _load_module(
    "core_contract_prompt_baseline_fixtures",
    "inner_os/evaluation/prompt_baseline_fixtures.py",
)

prompt_baselines_for_scenario = _BASELINE_MODULE.prompt_baselines_for_scenario


def build_core_contract_eval_summary() -> dict[str, Any]:
    scenario_results: list[dict[str, Any]] = []
    passed_count = 0
    for scenario_name in sorted(SCENARIOS.keys()):
        result = build_core_demo_result(scenario_name=scenario_name)
        evaluation = dict(result["evaluation"])
        if evaluation.get("passed"):
            passed_count += 1
        scenario_results.append(
            {
                "scenario_name": scenario_name,
                "description": result["scenario"]["description"],
                "input_text": result["scenario"]["input_text"],
                "expected_contract": result["expected_contract"],
                "reaction_contract": result["reaction_contract"],
                "evaluation": evaluation,
                "llm_expression_request": result["llm_expression_request"],
                "prompt_baselines": _build_prompt_baseline_results(scenario_name),
                "response_guideline": result["response_guideline"],
            }
        )

    total = len(scenario_results)
    return {
        "summary": {
            "scenario_count": total,
            "passed_count": passed_count,
            "failed_count": total - passed_count,
            "pass_rate": round((passed_count / total), 3) if total else 0.0,
        },
        "scenarios": scenario_results,
    }


def _build_prompt_baseline_results(scenario_name: str) -> list[dict[str, Any]]:
    expectation = CORE_QUICKSTART_EXPECTATIONS[scenario_name]
    baseline_results: list[dict[str, Any]] = []
    for sample in prompt_baselines_for_scenario(scenario_name):
        evaluation = evaluate_reaction_contract_against_expectation(
            reaction_contract=sample.observed_contract,
            expectation=expectation,
        )
        baseline_results.append(
            {
                "sample": sample.to_dict(),
                "evaluation": evaluation.to_dict(),
            }
        )
    return baseline_results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EQNet core quickstart の全シナリオを一括で評価する。",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="評価結果を JSON で出力する。",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    summary = build_core_contract_eval_summary()

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    print("EQNet Core Contract Evaluation")
    print("==============================")
    print(json.dumps(summary["summary"], ensure_ascii=False, indent=2))
    print()
    for scenario in summary["scenarios"]:
        evaluation = scenario["evaluation"]
        print(f"[{scenario['scenario_name']}]")
        print(f"説明: {scenario['description']}")
        print(f"入力: {scenario['input_text']}")
        print(f"passed: {evaluation['passed']}  score: {evaluation['score']}")
        if evaluation["violations"]:
            print("violations:")
            for violation in evaluation["violations"]:
                print(f"  - {violation['code']}: {violation['detail']}")
        if scenario["prompt_baselines"]:
            print("prompt baselines:")
            for baseline in scenario["prompt_baselines"]:
                sample = baseline["sample"]
                baseline_eval = baseline["evaluation"]
                print(
                    f"  - {sample['model_label']}: "
                    f"passed={baseline_eval['passed']} score={baseline_eval['score']}"
                )
        request = scenario["llm_expression_request"]
        print(
            "llm bridge: "
            f"should_call={request['should_call_llm']} "
            f"channel={request['action_channel']}"
        )
        print(f"guideline: {scenario['response_guideline']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
