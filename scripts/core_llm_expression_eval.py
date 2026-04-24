from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EMOT_ROOT = REPO_ROOT / "emot_terrain_lab"
if EMOT_ROOT.exists() and str(EMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(EMOT_ROOT))

from scripts.core_quickstart_demo import SCENARIOS, build_core_demo_result  # noqa: E402
from emot_terrain_lab.terrain import llm as terrain_llm  # noqa: E402


def _load_module(module_name: str, relative_path: str) -> Any:
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_BRIDGE_REVIEW_MODULE = _load_module(
    "core_llm_expression_bridge_contract",
    "inner_os/expression/llm_bridge_contract.py",
)

review_llm_bridge_text = _BRIDGE_REVIEW_MODULE.review_llm_bridge_text


def evaluate_core_llm_expression(
    *,
    scenario_name: str,
    text: str | None = None,
    temperature: float = 0.45,
    top_p: float | None = 0.9,
    call_llm: bool = True,
    model_label: str = "",
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
    if call_llm:
        started = time.perf_counter()
        raw_text = terrain_llm.chat_text(
            request["system_prompt"],
            request["user_prompt"],
            temperature=temperature,
            top_p=top_p,
        ) or ""
        latency_ms = (time.perf_counter() - started) * 1000.0

    review = review_llm_bridge_text(
        raw_text=raw_text,
        reaction_contract=request["contract"],
        fallback_text="",
    )
    return {
        "scenario_name": scenario_name,
        "run_metadata": run_metadata,
        "called_llm": bool(call_llm),
        "latency_ms": round(latency_ms, 4),
        "llm_expression_request": request,
        "raw_text": raw_text,
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


def _build_run_metadata(
    *,
    model_label: str,
    temperature: float,
    top_p: float | None,
    call_llm: bool,
) -> dict[str, Any]:
    return {
        "model_label": model_label or _default_model_label(),
        "temperature": temperature,
        "top_p": top_p,
        "call_llm": call_llm,
    }


def _default_model_label() -> str:
    return os.getenv("OPENAI_MODEL") or os.getenv("LMSTUDIO_MODEL") or "unconfigured"


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
        "--save-jsonl",
        default="",
        help="評価結果を JSONL で追記保存するパス。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="LLM を呼ばず、生成される request だけを確認する。",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    result = evaluate_core_llm_expression(
        scenario_name=args.scenario,
        text=args.text or None,
        temperature=args.temperature,
        top_p=args.top_p,
        call_llm=not args.dry_run,
        model_label=args.model_label,
    )
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
