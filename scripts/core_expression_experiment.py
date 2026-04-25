from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emot_terrain_lab.terrain import llm as terrain_llm  # noqa: E402
from inner_os.expression.llm_bridge_contract import review_llm_bridge_text  # noqa: E402
from inner_os.expression.speech_act_contract import SPEECH_ACT_LABELS  # noqa: E402
from scripts.baseline_router import load_router_config, route_baseline_prompt  # noqa: E402
from scripts.core_expression_eval_report import build_core_expression_eval_report  # noqa: E402
from scripts.core_llm_expression_eval import (  # noqa: E402
    evaluate_core_llm_expression,
)
from scripts.core_quickstart_demo import SCENARIOS, build_core_demo_result  # noqa: E402


BASELINE_NORMAL_SYSTEM_PROMPT = "Answer normally in natural Japanese."
BASELINE_PROMPT_SYSTEM_PROMPT = (
    "Respond in natural Japanese. Be empathetic, but do not over-interpret, "
    "do not over-advise, and keep the response concise."
)
DEFAULT_ROUTER_CONFIG_PATH = REPO_ROOT / "config" / "eval" / "baseline_router.yaml"


def run_core_expression_experiment(
    *,
    input_records: Iterable[Mapping[str, Any]],
    out_dir: str | Path,
    call_llm: bool = True,
    generator_model_label: str = "",
    generator_model: str = "",
    classifier_model_label: str = "",
    classifier_model: str = "",
    classify_output: bool = False,
    router_config_path: str | Path = DEFAULT_ROUTER_CONFIG_PATH,
) -> dict[str, Any]:
    output_dir = _resolve_path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [_normalize_case(record) for record in input_records]

    paths = {
        "input": output_dir / "input.jsonl",
        "speech_act_gold": output_dir / "speech_act_gold.jsonl",
        "baseline_normal": output_dir / "baseline_normal.jsonl",
        "baseline_prompt": output_dir / "baseline_prompt.jsonl",
        "baseline_router": output_dir / "baseline_router.jsonl",
        "eqnet": output_dir / "eqnet.jsonl",
        "baseline_normal_report": output_dir / "baseline_normal_contract_report.json",
        "baseline_prompt_report": output_dir / "baseline_prompt_contract_report.json",
        "baseline_router_report": output_dir / "baseline_router_contract_report.json",
        "eqnet_report": output_dir / "eqnet_contract_report.json",
        "readme": output_dir / "README.md",
    }
    _write_jsonl(paths["input"], cases)
    _write_jsonl(paths["speech_act_gold"], [_gold_record(case) for case in cases])

    resolved_generator_label = generator_model_label or generator_model
    resolved_classifier_label = classifier_model_label or classifier_model
    router_config = load_router_config(_resolve_path(router_config_path))
    with _temporary_model_env(generator_model):
        baseline_normal_records = [
            _run_baseline_case(
                case,
                mode="baseline_normal",
                system_prompt=BASELINE_NORMAL_SYSTEM_PROMPT,
                call_llm=call_llm,
                generator_model_label=resolved_generator_label,
            )
            for case in cases
        ]
        baseline_prompt_records = [
            _run_baseline_case(
                case,
                mode="baseline_prompt",
                system_prompt=BASELINE_PROMPT_SYSTEM_PROMPT,
                call_llm=call_llm,
                generator_model_label=resolved_generator_label,
            )
            for case in cases
        ]
        baseline_router_records = [
            _run_router_baseline_case(
                case,
                router_config=router_config,
                call_llm=call_llm,
                generator_model_label=resolved_generator_label,
            )
            for case in cases
        ]
        eqnet_records = [
            _run_eqnet_case(
                case,
                call_llm=call_llm,
                generator_model_label=resolved_generator_label,
                classifier_model_label=resolved_classifier_label,
                classifier_model=classifier_model,
                classify_output=classify_output,
            )
            for case in cases
        ]
    _write_jsonl(paths["baseline_normal"], baseline_normal_records)
    _write_jsonl(paths["baseline_prompt"], baseline_prompt_records)
    _write_jsonl(paths["baseline_router"], baseline_router_records)
    _write_jsonl(paths["eqnet"], eqnet_records)

    reports = {
        "baseline_normal": build_core_expression_eval_report(
            baseline_normal_records,
            group_by=("scenario_name", "generator_model_label", "response_channel"),
        ),
        "baseline_prompt": build_core_expression_eval_report(
            baseline_prompt_records,
            group_by=("scenario_name", "generator_model_label", "response_channel"),
        ),
        "baseline_router": build_core_expression_eval_report(
            baseline_router_records,
            group_by=(
                "scenario_name",
                "generator_model_label",
                "response_channel",
                "router_mode",
            ),
        ),
        "eqnet": build_core_expression_eval_report(
            eqnet_records,
            group_by=(
                "scenario_name",
                "generator_model_label",
                "classifier_model_label",
                "response_channel",
            ),
        ),
    }
    paths["baseline_normal_report"].write_text(
        json.dumps(reports["baseline_normal"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["baseline_prompt_report"].write_text(
        json.dumps(reports["baseline_prompt"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["baseline_router_report"].write_text(
        json.dumps(reports["baseline_router"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["eqnet_report"].write_text(
        json.dumps(reports["eqnet"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_readme(
        paths["readme"],
        case_count=len(cases),
        call_llm=call_llm,
        reports=reports,
    )
    return {
        "out_dir": str(output_dir),
        "case_count": len(cases),
        "router_config_path": str(_resolve_path(router_config_path)),
        "paths": {key: str(path) for key, path in paths.items()},
        "reports": reports,
    }


def _run_eqnet_case(
    case: Mapping[str, Any],
    *,
    call_llm: bool,
    generator_model_label: str,
    classifier_model_label: str,
    classifier_model: str,
    classify_output: bool,
) -> dict[str, Any]:
    result = evaluate_core_llm_expression(
        scenario_name=str(case["core_scenario"]),
        text=str(case["input"]),
        call_llm=call_llm,
        model_label=generator_model_label,
        classify_output=classify_output,
        classifier_model_label=classifier_model_label,
        classifier_model=classifier_model,
    )
    result["item_id"] = case["id"]
    result["scenario_name"] = case["scenario"]
    result["core_scenario"] = case["core_scenario"]
    result["input"] = case["input"]
    result["evaluation_mode"] = "eqnet"
    result["selected_response_channel"] = _selected_channel_from_final_action(result)
    result["expected_response_channel"] = case.get("expected_response_channel", "")
    return result


def _run_baseline_case(
    case: Mapping[str, Any],
    *,
    mode: str,
    system_prompt: str,
    call_llm: bool,
    generator_model_label: str,
) -> dict[str, Any]:
    core_result = build_core_demo_result(
        scenario_name=str(case["core_scenario"]),
        input_text=str(case["input"]),
    )
    request = core_result["llm_expression_request"]
    contract = request["contract"]
    raw_text = ""
    latency_ms = 0.0
    if call_llm:
        started = time.perf_counter()
        raw_text = terrain_llm.chat_text(
            system_prompt,
            str(case["input"]),
            temperature=0.45,
            top_p=0.9,
        ) or ""
        latency_ms = (time.perf_counter() - started) * 1000.0
    review = review_llm_bridge_text(
        raw_text=raw_text,
        reaction_contract=contract,
        fallback_text="",
    )
    final_action_type = "speak" if review.ok else "regenerate_or_review"
    if not call_llm and contract.get("response_channel") == "hold":
        final_action_type = "nonverbal"
    selected_response_channel = "speak" if call_llm else "hold"
    return {
        "item_id": case["id"],
        "scenario_name": case["scenario"],
        "core_scenario": case["core_scenario"],
        "input": case["input"],
        "evaluation_mode": mode,
        "selected_response_channel": selected_response_channel,
        "expected_response_channel": case.get("expected_response_channel", ""),
        "baseline_system_prompt": system_prompt,
        "called_llm": bool(call_llm),
        "latency_ms": round(latency_ms, 4),
        "run_metadata": {
            "model_label": generator_model_label or "unconfigured",
            "generator_model_label": generator_model_label or "unconfigured",
            "classifier_model_label": "not_used",
            "call_llm": call_llm,
            "classify_output": False,
        },
        "llm_expression_request": request,
        "raw_text": raw_text,
        "speech_act_analysis": None,
        "speech_act_analysis_error": "",
        "review": {
            "ok": review.ok,
            "raw_text": review.raw_text,
            "sanitized_text": review.sanitized_text,
            "violations": [
                {"code": violation.code, "detail": violation.detail}
                for violation in review.violations
            ],
        },
        "final_action": {
            "type": final_action_type,
            "text": review.sanitized_text,
        },
    }


def _run_router_baseline_case(
    case: Mapping[str, Any],
    *,
    router_config: Mapping[str, Any],
    call_llm: bool,
    generator_model_label: str,
) -> dict[str, Any]:
    decision = route_baseline_prompt(str(case["input"]), router_config)
    record = _run_baseline_case(
        case,
        mode="baseline_router",
        system_prompt=decision.prompt,
        call_llm=call_llm and decision.should_call_llm,
        generator_model_label=generator_model_label,
    )
    record["router_mode"] = decision.mode
    record["router_rule_name"] = decision.rule_name
    record["router_should_call_llm"] = decision.should_call_llm
    record["router_constraints"] = decision.constraints
    record["selected_response_channel"] = "speak" if decision.should_call_llm else "hold"
    record["run_metadata"]["router_mode"] = decision.mode
    record["run_metadata"]["router_rule_name"] = decision.rule_name
    record["run_metadata"]["router_should_call_llm"] = decision.should_call_llm
    record["run_metadata"]["selected_response_channel"] = record["selected_response_channel"]
    if not decision.should_call_llm:
        record["final_action"] = {
            "type": decision.final_action_type,
            "text": decision.fixed_text,
        }
    return record


def _selected_channel_from_final_action(record: Mapping[str, Any]) -> str:
    final_action = record.get("final_action")
    if isinstance(final_action, Mapping):
        action_type = str(final_action.get("type") or "")
        if action_type == "nonverbal":
            return "hold"
    request = record.get("llm_expression_request")
    if isinstance(request, Mapping):
        contract = request.get("contract")
        if isinstance(contract, Mapping):
            return str(contract.get("response_channel") or "")
    return ""


def _normalize_case(record: Mapping[str, Any]) -> dict[str, str]:
    item_id = str(record.get("id") or record.get("item_id") or "").strip()
    if not item_id:
        raise ValueError("input record requires id or item_id")
    input_text = str(record.get("input") or record.get("text") or "").strip()
    if not input_text:
        raise ValueError(f"input record {item_id} requires input")
    core_scenario = str(record.get("core_scenario") or "").strip()
    if core_scenario not in SCENARIOS:
        core_scenario = _infer_core_scenario(record)
    return {
        "id": item_id,
        "scenario": str(record.get("scenario") or core_scenario or "unknown"),
        "core_scenario": core_scenario,
        "input": input_text,
        "gold_speech_act": str(record.get("gold_speech_act") or "other"),
        "expected_response_channel": str(record.get("expected_response_channel") or ""),
    }


def _infer_core_scenario(record: Mapping[str, Any]) -> str:
    scenario = str(record.get("scenario") or "").lower()
    gold = str(record.get("gold_speech_act") or "").lower()
    if "withdraw" in scenario or gold == "hold":
        return "guarded_uncertainty"
    if "vent" in scenario or "advice" in scenario:
        return "guarded_uncertainty"
    return "small_shared_moment"


def _gold_record(case: Mapping[str, str]) -> dict[str, Any]:
    label = case["gold_speech_act"]
    labels = [label] if label in SPEECH_ACT_LABELS else ["other"]
    return {
        "item_id": case["id"],
        "scenario_name": case["scenario"],
        "raw_text": case["input"],
        "source": "human_gold_seed",
        "sentences": [
            {
                "text": case["input"],
                "labels": labels,
                "confidence": 1.0,
            }
        ],
    }


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _write_readme(
    path: Path,
    *,
    case_count: int,
    call_llm: bool,
    reports: Mapping[str, Mapping[str, Any]],
) -> None:
    lines = [
        "# Core Expression Eval Run 001",
        "",
        f"- case_count: {case_count}",
        f"- call_llm: {call_llm}",
        "",
        "## Summary",
        "",
    ]
    for name, report in reports.items():
        summary = report["summary"]
        lines.append(
            f"- {name}: violation_rate={summary['violation_rate']} "
            f"violations={summary['violation_codes']}"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- input.jsonl",
            "- speech_act_gold.jsonl",
            "- baseline_normal.jsonl",
            "- baseline_prompt.jsonl",
            "- baseline_router.jsonl",
            "- eqnet.jsonl",
            "- *_contract_report.json",
            "",
            "This run is for failure-mode measurement, not for proving human preference.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _resolve_path(path: str | Path) -> Path:
    output_path = Path(path)
    return output_path if output_path.is_absolute() else REPO_ROOT / output_path


@contextmanager
def _temporary_model_env(model: str) -> Iterator[None]:
    if not model:
        yield
        return
    keys = ("OPENAI_MODEL", "LMSTUDIO_MODEL")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = model
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_input_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = _resolve_path(path)
    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                payload = json.loads(stripped)
                if isinstance(payload, dict):
                    records.append(payload)
    return records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run baseline vs EQNet core expression evaluation package.",
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--generator-model-label", default="")
    parser.add_argument("--generator-model", default="")
    parser.add_argument("--classifier-model-label", default="")
    parser.add_argument("--classifier-model", default="")
    parser.add_argument("--router-config", default=str(DEFAULT_ROUTER_CONFIG_PATH))
    parser.add_argument("--classify-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    result = run_core_expression_experiment(
        input_records=load_input_jsonl(args.input_jsonl),
        out_dir=args.out_dir,
        call_llm=not args.dry_run,
        generator_model_label=args.generator_model_label,
        generator_model=args.generator_model,
        classifier_model_label=args.classifier_model_label,
        classifier_model=args.classifier_model,
        classify_output=args.classify_output,
        router_config_path=args.router_config,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print("Core Expression Experiment")
    print("==========================")
    print(f"out_dir: {result['out_dir']}")
    print(f"case_count: {result['case_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
