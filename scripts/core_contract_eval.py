from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.core_quickstart_demo import SCENARIOS, build_core_demo_result


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
        print(f"guideline: {scenario['response_guideline']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
