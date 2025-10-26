#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick CLI to adjust persona user preferences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from persona import PersonaState, compose_controls
from persona.learner import feedback_from_labels, update_preferences

USER_PREF_PATH = Path("state/persona_user_pref.json")


def load_user_pref() -> Dict[str, float]:
    if USER_PREF_PATH.exists():
        try:
            return json.loads(USER_PREF_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_user_pref(pref: Dict[str, float]) -> None:
    USER_PREF_PATH.parent.mkdir(parents=True, exist_ok=True)
    USER_PREF_PATH.write_text(json.dumps(pref, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Persona calibration helper")
    ap.add_argument("--culture", default="anime_2010s_slice")
    ap.add_argument("--mode", default="caregiver")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument(
        "--feedback",
        action="append",
        default=[],
        help="Feedback labels such as more_direct, softer, speed_up",
    )
    ap.add_argument("--snapshot", help="Save composition snapshot label")
    args = ap.parse_args()

    pref = load_user_pref()
    for label in args.feedback:
        delta = feedback_from_labels(label)
        if delta:
            pref = update_preferences(pref, delta, lr=0.2)
    save_user_pref(pref)

    composition = compose_controls(
        culture_name=args.culture,
        mode_name=args.mode,
        user_pref=pref,
        alpha=args.alpha if args.alpha is not None else pref.get("__alpha", 0.4),
        beta=args.beta,
        safety={"beta": args.beta},
    )

    print("=== Persona Style Preview ===")
    print(json.dumps(composition.to_dict(), ensure_ascii=False, indent=2))

    if args.snapshot:
        state = PersonaState()
        state.save_snapshot(args.snapshot, composition.to_dict())
        print(f"Snapshot saved to {args.snapshot}")


if __name__ == "__main__":
    main()
