from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EMOT_ROOT = REPO_ROOT / "emot_terrain_lab"
if EMOT_ROOT.exists() and str(EMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(EMOT_ROOT))

from scripts.core_quickstart_demo import SCENARIOS, build_core_demo_result  # noqa: E402
from emot_terrain_lab.terrain import llm as terrain_llm  # noqa: E402
from inner_os.expression.llm_bridge_contract import review_llm_bridge_text  # noqa: E402
from inner_os.expression.speech_act_contract import (  # noqa: E402
    build_speech_act_classification_request,
    speech_act_analysis_from_dict,
)


def evaluate_core_llm_expression(
    *,
    scenario_name: str,
    text: str | None = None,
    temperature: float = 0.45,
    top_p: float | None = 0.9,
    call_llm: bool = True,
    model_label: str = "",
    classify_output: bool = False,
    classifier_model_label: str = "",
    classifier_model: str = "",
    classifier_base_url: str = "",
    classifier_api_key: str = "",
    speech_act_analysis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = build_core_demo_result(
        scenario_name=scenario_name,
        input_text=text or None,
    )
    request = result["llm_expression_request"]
    run_metadata = _build_run_metadata(
        model_label=model_label,
        temperature=temperature,
        top_p=top_p,
        call_llm=call_llm,
        classify_output=classify_output,
        classifier_model_label=classifier_model_label,
        classifier_model=classifier_model,
        classifier_base_url=classifier_base_url,
        has_external_speech_act_analysis=speech_act_analysis is not None,
    )
    if not request["should_call_llm"]:
        return {
            "scenario_name": scenario_name,
            "run_metadata": run_metadata,
            "called_llm": False,
            "skip_reason": request["blocked_reason"],
            "llm_expression_request": request,
            "review": {
                "ok": True,
                "raw_text": "",
                "sanitized_text": "",
                "violations": [],
            },
            "final_action": request["fallback_action"],
        }

    raw_text = ""
    latency_ms = 0.0
    resolved_speech_act_analysis: dict[str, Any] | None = (
        speech_act_analysis_from_dict(speech_act_analysis).to_dict()
        if isinstance(speech_act_analysis, Mapping)
        else None
    )
    speech_act_analysis_error = ""
    if call_llm:
        started = time.perf_counter()
        raw_text = terrain_llm.chat_text(
            request["system_prompt"],
            request["user_prompt"],
            temperature=temperature,
            top_p=top_p,
        ) or ""
        latency_ms = (time.perf_counter() - started) * 1000.0
        if classify_output and raw_text.strip() and resolved_speech_act_analysis is None:
            try:
                classification_request = build_speech_act_classification_request(raw_text)
                with _temporary_classifier_env(
                    model=classifier_model,
                    base_url=classifier_base_url,
                    api_key=classifier_api_key,
                ):
                    classifier_raw = terrain_llm.chat_text(
                        classification_request["system_prompt"],
                        classification_request["user_prompt"],
                        temperature=0.0,
                        top_p=1.0,
                    ) or ""
                parsed = json.loads(_extract_json_object(classifier_raw))
                resolved_speech_act_analysis = speech_act_analysis_from_dict(parsed).to_dict()
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                speech_act_analysis_error = str(exc)

    review = review_llm_bridge_text(
        raw_text=raw_text,
        reaction_contract=request["contract"],
        fallback_text="",
        speech_act_analysis=resolved_speech_act_analysis,
    )
    return {
        "scenario_name": scenario_name,
        "run_metadata": run_metadata,
        "called_llm": bool(call_llm),
        "latency_ms": round(latency_ms, 4),
        "llm_expression_request": request,
        "raw_text": raw_text,
        "speech_act_analysis": resolved_speech_act_analysis,
        "speech_act_analysis_error": speech_act_analysis_error,
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
            "type": "speak" if review.ok else "regenerate_or_review",
            "text": review.sanitized_text,
        },
    }


def save_eval_jsonl(path: str | Path, record: dict[str, Any]) -> Path:
    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
    return output_path


def load_speech_act_analysis_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))
    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def find_speech_act_analysis(
    records: list[dict[str, Any]],
    *,
    scenario_name: str,
    raw_text: str,
) -> dict[str, Any] | None:
    for record in records:
        if str(record.get("scenario_name") or "") not in {"", scenario_name}:
            continue
        candidate_text = str(record.get("raw_text") or record.get("text") or "")
        if candidate_text and candidate_text != raw_text:
            continue
        analysis = record.get("speech_act_analysis")
        if isinstance(analysis, Mapping):
            return dict(analysis)
        if isinstance(record.get("sentences"), list):
            return {
                "schema_version": record.get("schema_version") or "speech_act.v1",
                "source": record.get("source") or "jsonl_label",
                "sentences": record["sentences"],
            }
    return None


def _build_run_metadata(
    *,
    model_label: str,
    temperature: float,
    top_p: float | None,
    call_llm: bool,
    classify_output: bool,
    classifier_model_label: str,
    classifier_model: str,
    classifier_base_url: str,
    has_external_speech_act_analysis: bool,
) -> dict[str, Any]:
    generator_model_label = model_label or _default_model_label()
    resolved_classifier_model_label = (
        classifier_model_label
        or classifier_model
        or (generator_model_label if classify_output else "")
        or "not_used"
    )
    return {
        "model_label": generator_model_label,
        "generator_model_label": generator_model_label,
        "classifier_model_label": resolved_classifier_model_label,
        "classifier_base_url": classifier_base_url or "",
        "temperature": temperature,
        "top_p": top_p,
        "call_llm": call_llm,
        "classify_output": classify_output,
        "has_external_speech_act_analysis": has_external_speech_act_analysis,
    }


def _default_model_label() -> str:
    return os.getenv("OPENAI_MODEL") or os.getenv("LMSTUDIO_MODEL") or "unconfigured"


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("speech-act classifier did not return a JSON object")
    return stripped[start : end + 1]


@contextmanager
def _temporary_classifier_env(
    *,
    model: str,
    base_url: str,
    api_key: str,
) -> Iterator[None]:
    overrides = {
        "OPENAI_MODEL": model,
        "LMSTUDIO_MODEL": model,
        "OPENAI_BASE_URL": base_url,
        "OPENAI_API_KEY": api_key,
    }
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="core quickstart の state-conditioned LLM expression を評価する。",
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS.keys()),
        default="small_shared_moment",
    )
    parser.add_argument("--text", default="")
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--model-label",
        default="",
        help="評価ログに残すモデル名。未指定時は OPENAI_MODEL / LMSTUDIO_MODEL を使う。",
    )
    parser.add_argument(
        "--classifier-model-label",
        default="",
        help="分類器モデル名を評価ログに残す。未指定時は --classifier-model を使う。",
    )
    parser.add_argument(
        "--classifier-model",
        default="",
        help="speech-act 分類だけに使う OPENAI_MODEL override。",
    )
    parser.add_argument(
        "--classifier-base-url",
        default="",
        help="speech-act 分類だけに使う OPENAI_BASE_URL override。",
    )
    parser.add_argument(
        "--classifier-api-key",
        default="",
        help="speech-act 分類だけに使う OPENAI_API_KEY override。",
    )
    parser.add_argument(
        "--speech-act-jsonl",
        default="",
        help="人手ラベル済み speech-act JSONL を使って contract review する。",
    )
    parser.add_argument(
        "--save-jsonl",
        default="",
        help="評価結果を JSONL で追記保存するパス。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="LLM を呼ばず、生成される request だけを確認する。",
    )
    parser.add_argument(
        "--classify-output",
        action="store_true",
        help="Run a speech-act classifier before contract review.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    speech_act_records = (
        load_speech_act_analysis_jsonl(args.speech_act_jsonl)
        if args.speech_act_jsonl
        else []
    )
    external_speech_act_analysis = None
    if speech_act_records and args.dry_run:
        external_speech_act_analysis = find_speech_act_analysis(
            speech_act_records,
            scenario_name=args.scenario,
            raw_text="",
        )
    result = evaluate_core_llm_expression(
        scenario_name=args.scenario,
        text=args.text or None,
        temperature=args.temperature,
        top_p=args.top_p,
        call_llm=not args.dry_run,
        model_label=args.model_label,
        classify_output=args.classify_output,
        classifier_model_label=args.classifier_model_label,
        classifier_model=args.classifier_model,
        classifier_base_url=args.classifier_base_url,
        classifier_api_key=args.classifier_api_key,
        speech_act_analysis=external_speech_act_analysis,
    )
    if speech_act_records and not external_speech_act_analysis and result.get("raw_text"):
        matched_analysis = find_speech_act_analysis(
            speech_act_records,
            scenario_name=args.scenario,
            raw_text=str(result.get("raw_text") or ""),
        )
        if matched_analysis is not None:
            review = review_llm_bridge_text(
                raw_text=str(result.get("raw_text") or ""),
                reaction_contract=result["llm_expression_request"]["contract"],
                fallback_text="",
                speech_act_analysis=matched_analysis,
            )
            result["speech_act_analysis"] = speech_act_analysis_from_dict(
                matched_analysis
            ).to_dict()
            result["speech_act_analysis_error"] = ""
            result["run_metadata"]["has_external_speech_act_analysis"] = True
            result["review"] = {
                "ok": review.ok,
                "raw_text": review.raw_text,
                "sanitized_text": review.sanitized_text,
                "violations": [
                    {"code": violation.code, "detail": violation.detail}
                    for violation in review.violations
                ],
            }
            result["final_action"] = {
                "type": "speak" if review.ok else "regenerate_or_review",
                "text": review.sanitized_text,
            }
    if args.save_jsonl:
        saved_path = save_eval_jsonl(args.save_jsonl, result)
        result["saved_jsonl"] = str(saved_path)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print("EQNet Core LLM Expression Eval")
    print("==============================")
    print(f"scenario: {result['scenario_name']}")
    print(f"model_label: {result['run_metadata']['model_label']}")
    print(f"called_llm: {result['called_llm']}")
    print(f"classify_output: {result['run_metadata']['classify_output']}")
    if "skip_reason" in result:
        print(f"skip_reason: {result['skip_reason']}")
    print(f"review_ok: {result['review']['ok']}")
    if result["review"]["violations"]:
        print("violations:")
        for violation in result["review"]["violations"]:
            print(f"  - {violation['code']}: {violation['detail']}")
    if result.get("raw_text"):
        print()
        print("[raw_text]")
        print(result["raw_text"])
    print()
    print("[final_action]")
    print(json.dumps(result["final_action"], ensure_ascii=False, indent=2))
    if result.get("saved_jsonl"):
        print(f"saved_jsonl: {result['saved_jsonl']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
