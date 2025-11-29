"""Minimal demo that spins up personas and prints their life indicators."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Callable, List

from eqnet.hub.multi_tenant import EQNetHubManager
from eqnet.persona.loader import load_persona_from_dir


def dummy_embed(text: str) -> List[float]:
    """Demo-only embedding stub (replace with a real encoder)."""

    base = float(len(text) % 7) / 6.0
    return [base, 0.1, 0.2]


def discover_persona_ids(persona_dir: Path) -> List[str]:
    """Return persona IDs discovered from *.yaml under persona_dir."""

    if not persona_dir.exists():
        return []
    return sorted(p.stem for p in persona_dir.glob("*.yaml"))


def demo_text_from_persona(user_id: str, persona_dir: Path, fallback: str) -> str:
    persona = load_persona_from_dir(persona_dir, user_id)
    if persona is not None:
        speech = persona.speech or {}
        demo_text = speech.get("demo_text")
        if demo_text:
            return str(demo_text)
        demo_cfg = persona.raw.get("demo") if isinstance(persona.raw, dict) else {}
        if isinstance(demo_cfg, dict) and demo_cfg.get("sample_text"):
            return str(demo_cfg.get("sample_text"))
        diary_style = persona.diary_style or {}
        sample_entry = diary_style.get("sample_text")
        if sample_entry:
            return str(sample_entry)
    return fallback


def resolve_embed_fn(spec: str) -> Callable[[str], List[float]]:
    if spec == "dummy":
        return dummy_embed
    if ":" not in spec:
        raise ValueError("--embed-fn must be 'dummy' or 'module:function'")
    module_name, func_name = spec.split(":", 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, func_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EQNet demo for one or more personas")
    parser.add_argument("--base-dir", type=Path, default=Path("eqnet_data"), help="Where EQNet state/logs are stored")
    parser.add_argument("--persona-dir", type=Path, default=Path("personas"), help="Directory containing persona YAMLs")
    parser.add_argument("--user", action="append", dest="users", help="Persona IDs to run (defaults to auto discovery)")
    parser.add_argument(
        "--fallback-text",
        default="今日は新しい一歩を踏み出せた気がする。",
        help="Default text when persona does not define demo_text/sample_text",
    )
    parser.add_argument(
        "--embed-fn",
        default="dummy",
        help="Text embedding function spec (dummy or module:function)",
    )
    parser.add_argument(
        "--event-type",
        default="demo",
        help='Value stored in raw_event["type"]. Examples: demo/chat/live',
    )
    args = parser.parse_args()

    embed_fn = resolve_embed_fn(args.embed_fn)
    manager = EQNetHubManager(args.base_dir, embed_text_fn=embed_fn, persona_dir=args.persona_dir)

    user_ids = args.users or discover_persona_ids(args.persona_dir)
    if not user_ids:
        print(f"No personas found in {args.persona_dir}.")
        return

    for user_id in user_ids:
        hub = manager.for_user(user_id)
        text = demo_text_from_persona(user_id, args.persona_dir, args.fallback_text)
        hub.log_moment(raw_event={"type": args.event_type}, raw_text=text)
        hub.run_nightly()
        state = hub.query_state()
        persona_meta = state.get("persona", {}) or {}
        display_name = persona_meta.get("display_name", user_id)
        print(f"=== {display_name} ({user_id}) ===")
        print("  input_text:", text)
        print("  LifeIndicator:", state.get("life_indicator"))
        print("  PolicyPrior:", state.get("policy_prior"))
        print()


if __name__ == "__main__":
    main()
