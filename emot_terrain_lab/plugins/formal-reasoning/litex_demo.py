"""Litex rewrite skeleton for EQNet control tuning.

This script demonstrates how EQNet could call into ``pylitex`` to transform
affect/control expressions before handing them to Lean for verification.

It is safe to run even when ``pylitex`` is not installed; in that case the
script prints guidance instead of failing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


TRY_RULES = [
    "pause_relax",
    "warmth_soften",
    "micro_affirmation",
    "sigma_dwell",
]

EXPR = "sigma.high + love_mode.warm -> warmth_up + pause_down + gaze_hold"


def run_demo() -> None:
    try:
        from pylitex import Rewriter, parse_expr  # type: ignore
    except ImportError:
        print("⚠️  pylitex is not installed. Install with `pip install pylitex`.")
        print("When available, the demo will rewrite expression trees like:")
        print(f"  {EXPR}")
        return

    rw = Rewriter(TRY_RULES)
    ast = parse_expr(EXPR)
    nf, steps = rw.normalize(ast, return_steps=True)

    print("🔁 Litex rewrite steps:")
    for step in steps:
        print("  •", step)

    print("\n✅ Normal form:")
    print("  ", nf)


def main(argv: list[str] | None = None) -> int:
    run_demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
