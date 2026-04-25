from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.core_quickstart_demo import build_core_demo_result


DEFAULT_INPUT_PATH = REPO_ROOT / "config" / "eval" / "context_axis_contrast_cases.json"


def run_context_axis_contrast(
    *,
    input_path: str | Path = DEFAULT_INPUT_PATH,
) -> dict[str, Any]:
    payload = _load_json(input_path)
    default_core_scenario = str(payload.get("default_core_scenario") or "small_shared_moment")
    rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    for case in _list_of_mappings(payload.get("cases")):
        case_id = str(case.get("id") or "").strip()
        input_text = str(case.get("input") or "").strip()
        core_scenario = str(case.get("core_scenario") or default_core_scenario)
        variants = _mapping(case.get("variants"))
        for variant_name, variant_payload in variants.items():
            variant = _mapping(variant_payload)
            context = _mapping(variant.get("expression_context_state"))
            expected = _mapping(variant.get("expected"))
            result = build_core_demo_result(
                scenario_name=core_scenario,
                input_text=input_text,
                expression_context_state=context,
            )
            contract = dict(result["reaction_contract"])
            surface_policy = dict(result["llm_expression_request"]["surface_policy"])
            row = {
                "case_id": case_id,
                "variant": str(variant_name),
                "core_scenario": core_scenario,
                "input": input_text,
                "expected": expected,
                "contract": _contract_view(contract),
                "surface_policy": _surface_policy_view(surface_policy),
                "audit_axes": dict(result["quick_audit_projection"]["audit_axes"]),
            }
            row["passed"] = _matches_expected(row["contract"], expected)
            if not row["passed"]:
                mismatches.append(row)
            rows.append(row)

    return {
        "schema_version": "context_axis_contrast.v1",
        "input_path": str(_resolve_path(input_path)),
        "case_count": len(_list_of_mappings(payload.get("cases"))),
        "variant_count": len(rows),
        "passed": not mismatches,
        "mismatch_count": len(mismatches),
        "rows": rows,
    }


def write_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    output_path = _resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def _contract_view(contract: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "response_channel",
        "stance",
        "scale",
        "initiative",
        "shape_id",
        "strategy",
        "execution_mode",
        "interpretation_budget",
        "question_budget",
        "timing_mode",
        "distance_mode",
        "closure_mode",
    )
    return {key: contract.get(key) for key in keys}


def _surface_policy_view(surface_policy: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "response_channel",
        "max_sentences",
        "question_budget",
        "interpretation_budget",
        "advice_budget",
        "brightness_budget",
        "fallback_shape_id",
    )
    return {key: surface_policy.get(key) for key in keys}


def _matches_expected(contract: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(contract.get(key) == value for key, value in expected.items())


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_resolve_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("contrast input must contain a JSON object")
    return payload


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_of_mappings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run same-input context-axis contrast checks against EQNet quick core.",
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output", default="")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    report = run_context_axis_contrast(input_path=args.input)
    if args.output:
        write_report(args.output, report)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"context contrast variants={report['variant_count']} "
            f"mismatches={report['mismatch_count']}"
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
