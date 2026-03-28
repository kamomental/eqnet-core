#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LM Studio 実機で EQNet パイプラインを確認するスクリプト。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
EMOT_ROOT = REPO_ROOT / "emot_terrain_lab"
if EMOT_ROOT.exists() and str(EMOT_ROOT) not in sys.path:
    sys.path.append(str(EMOT_ROOT))

from emot_terrain_lab.hub import EmotionalHubRuntime, RuntimeConfig
from emot_terrain_lab.hub.lmstudio_pipeline_probe import (
    build_lmstudio_pipeline_probe,
    render_lmstudio_pipeline_probe,
)
from emot_terrain_lab.terrain import llm as terrain_llm


def load_env() -> None:
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv(override=True)


def apply_model_override(model_name: str) -> None:
    normalized = str(model_name or "").strip()
    if normalized:
        os.environ["LMSTUDIO_MODEL"] = normalized


def plain_lmstudio_response(prompt: str) -> dict[str, Any]:
    start = time.perf_counter()
    text = terrain_llm.chat_text(
        "You are a neutral assistant. Respond directly and briefly.",
        prompt,
        temperature=0.6,
        top_p=0.95,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    llm_info = terrain_llm.get_llm()
    return {
        "text": text,
        "latency_ms": round(float(latency_ms), 4),
        "model": str(getattr(llm_info, "model", "") or ""),
        "model_source": str(getattr(llm_info, "model_source", "") or ""),
        "base_url": str(getattr(llm_info, "base_url", "") or ""),
        "available": bool(getattr(llm_info, "available", False)),
    }


def run_eqnet_pipeline(
    *,
    prompt: str,
    context: str,
    intent: str,
    fast_only: bool,
    force_llm_bridge: bool,
    history: list[str],
) -> dict[str, Any]:
    runtime = EmotionalHubRuntime(
        RuntimeConfig(
            use_eqnet_core=True,
            force_llm_bridge=force_llm_bridge,
        )
    )
    runtime._surface_response_history.extend(
        str(item or "").strip() for item in (history or []) if str(item or "").strip()
    )
    result = runtime.process_turn(
        user_text=prompt,
        context=context or None,
        intent=intent or None,
        fast_only=fast_only,
    )
    probe = build_lmstudio_pipeline_probe(
        result,
        current_text=prompt,
        history=history,
    )
    return {
        "result": result,
        "probe": probe,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="LM Studio と EQNet パイプラインを並べて確認します。")
    parser.add_argument(
        "--prompt",
        "-p",
        default="少ししんどいけれど、何から話せばいいかわからないです。",
        help="確認したいユーザー入力",
    )
    parser.add_argument(
        "--context",
        default="",
        help="追加の文脈",
    )
    parser.add_argument(
        "--intent",
        default="check_in",
        help="runtime に渡す intent",
    )
    parser.add_argument(
        "--history",
        action="append",
        default=[],
        help="直近履歴。複数回指定できます。",
    )
    parser.add_argument(
        "--fast-only",
        action="store_true",
        help="fast_only で実行します。",
    )
    parser.add_argument(
        "--force-llm-bridge",
        action="store_true",
        help="habit/reflex の近道を抑えて LM Studio bridge を確認します。",
    )
    parser.add_argument(
        "--skip-plain",
        action="store_true",
        help="plain LM Studio 応答の比較を省略します。",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="JSON で出力します。",
    )
    parser.add_argument(
        "--model",
        default="",
        help="LM Studio で明示的に使うモデル名",
    )
    args = parser.parse_args()

    load_env()
    apply_model_override(args.model)

    plain_payload: dict[str, Any] | None = None
    if not args.skip_plain:
        plain_payload = plain_lmstudio_response(args.prompt)

    eqnet_payload = run_eqnet_pipeline(
        prompt=args.prompt,
        context=args.context,
        intent=args.intent,
        fast_only=bool(args.fast_only),
        force_llm_bridge=bool(args.force_llm_bridge),
        history=list(args.history or []),
    )
    probe = eqnet_payload["probe"]

    if args.json:
        payload = {
            "plain_lmstudio": plain_payload,
            "eqnet_probe": probe.to_dict(),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if plain_payload is not None:
        print("## Plain LM Studio")
        print(
            f"- model: {plain_payload.get('model') or '(unknown)'}"
            f" / source: {plain_payload.get('model_source') or '(unknown)'}"
        )
        print(f"- latency_ms: {plain_payload.get('latency_ms')}")
        print(plain_payload.get("text") or "(no response)")
        print("")

    print(render_lmstudio_pipeline_probe(probe))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
