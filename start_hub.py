#!/usr/bin/env python
"""CLI entry-point for the EQNet hub runtime."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ROOT_PKG = ROOT / "emot_terrain_lab"
if ROOT_PKG.exists():
    pkg_path = str(ROOT_PKG)
    if pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)


def _load_runtime_classes() -> tuple[type[Any], type[Any]]:
    module = importlib.import_module("emot_terrain_lab.hub.runtime")
    return module.EmotionalHubRuntime, module.RuntimeConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the EmotionalHubRuntime as the canonical CLI runtime."
    )
    parser.add_argument(
        "--state-dir",
        default="data/state_hub",
        help="Directory used by the hub runtime for persistent EQNet state.",
    )
    parser.add_argument(
        "--context",
        default=None,
        help="Optional context text passed into every turn.",
    )
    parser.add_argument(
        "--intent",
        default=None,
        help="Optional intent label routed to the hub.",
    )
    parser.add_argument(
        "--once",
        default=None,
        help="Run a single turn and exit.",
    )
    parser.add_argument(
        "--fast-only",
        action="store_true",
        help="Skip heavier recall/LLM paths and emit only the fast acknowledgement path.",
    )
    parser.add_argument(
        "--no-eqnet-core",
        action="store_true",
        help="Disable EQNet core integration for debugging.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full turn payload as JSON in --once mode.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Optional image path passed to the LM Studio VLM adapter.",
    )
    return parser


def _serialize_result(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return dict(result)


def _run_turn(
    runtime: Any,
    *,
    user_text: str,
    context: Optional[str],
    intent: Optional[str],
    fast_only: bool,
    image_path: Optional[str],
) -> dict[str, Any]:
    return runtime.process_turn(
        user_text=user_text,
        context=context,
        intent=intent,
        fast_only=fast_only,
        image_path=image_path,
    )


def _print_turn_summary(result: dict[str, Any]) -> None:
    payload = _serialize_result(result)
    response = payload.get("response") or {}
    text = response.get("text")
    print(f"talk_mode={payload.get('talk_mode')}")
    print(f"response_route={payload.get('response_route')}")
    if text:
        print(text)
    else:
        print("[no response]")



def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    EmotionalHubRuntime, RuntimeConfig = _load_runtime_classes()
    runtime = EmotionalHubRuntime(
        RuntimeConfig(
            use_eqnet_core=not args.no_eqnet_core,
            eqnet_state_dir=args.state_dir,
        )
    )

    if args.once is not None:
        result = _run_turn(
            runtime,
            user_text=args.once,
            context=args.context,
            intent=args.intent,
            fast_only=args.fast_only,
            image_path=args.image,
        )
        if args.json:
            print(json.dumps(_serialize_result(result), ensure_ascii=False, indent=2))
        else:
            _print_turn_summary(result)
        return 0

    print("EQNet hub runtime CLI")
    print("Type /quit to exit.")
    return runtime.run_forever(
        context=args.context,
        intent=args.intent,
        fast_only=args.fast_only,
        image_path=args.image,
        render_fn=_print_turn_summary,
    )


if __name__ == "__main__":
    raise SystemExit(main())
