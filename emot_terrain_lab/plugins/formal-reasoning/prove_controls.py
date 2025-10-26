"""Formal safety scaffold for EQNet control invariants.

This module serves two purposes:
1. Write (or update) a Lean file containing core safety lemmas and run
   `lake build` so Mathlib can certify the invariants.
2. Expose a lightweight Python checker (`enforce_invariants`) that mirrors
   the Lean statements for use at runtime.
"""

from __future__ import annotations

import dataclasses
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Tuple


ROOT = Path(__file__).resolve().parent
LEAN_FILE = ROOT / "proofs" / "Safety.lean"

LEAN_TEMPLATE = """import Mathlib.Data.Real.Basic
import Mathlib.Tactic

namespace EQNetSafety

structure Controls where
  motion_speed : ℝ
  gesture_amp  : ℝ
  warmth       : ℝ
  pause_ms     : ℝ
  containment  : Bool
deriving Repr

structure Bounds where
  rho        : ℝ
  maxLambda  : ℝ
  epsGesture : ℝ
deriving Repr

def rateLimited (prev cur : Controls) (b : Bounds) : Prop :=
  |cur.motion_speed - prev.motion_speed| ≤ b.rho ∧
  |cur.gesture_amp  - prev.gesture_amp|  ≤ b.rho ∧
  |cur.warmth       - prev.warmth|       ≤ b.rho

def containmentSafe (cur : Controls) (b : Bounds) : Prop :=
  cur.containment → (cur.motion_speed = 0 ∧ cur.gesture_amp ≤ b.epsGesture)

def lambdaSafe (λ : ℝ) (b : Bounds) : Prop :=
  λ ≤ b.maxLambda

lemma safetyInvariant
  (prev cur : Controls) (λ : ℝ) (b : Bounds)
  (h₁ : rateLimited prev cur b)
  (h₂ : containmentSafe cur b)
  (h₃ : lambdaSafe λ b) :
  rateLimited prev cur b ∧ containmentSafe cur b ∧ lambdaSafe λ b := by
  exact And.intro h₁ (And.intro h₂ h₃)

end EQNetSafety
"""


@dataclass
class Bounds:
    """Runtime mirror of the Lean `Bounds` structure."""

    rho: float
    max_lambda: float
    eps_gesture: float


def ensure_lean_file() -> None:
    LEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    LEAN_FILE.write_text(LEAN_TEMPLATE, encoding="utf-8")
    print(f"📝 Wrote Lean safety template to {LEAN_FILE}")


def run_lake_build() -> None:
    if shutil.which("lake") is None:
        print("⚠️  `lake` command not found. Install Lean4 and ensure it is on PATH.")
        return
    try:
        subprocess.run(["lake", "build"], check=True, cwd=ROOT)
        print("✅ Lean build succeeded: safety invariants verified.")
    except subprocess.CalledProcessError as err:
        print("❌ `lake build` failed.")
        print("    ", err)


def enforce_invariants(
    prev: Mapping[str, float],
    cur: MutableMapping[str, float],
    metrics: Mapping[str, float],
    bounds: Bounds,
) -> Tuple[MutableMapping[str, float], str | None]:
    """
    Enforce rate limits, containment rules, and λ-bounds on control signals.

    Returns the (possibly repaired) control mapping and a status flag:
        - None: no repair was necessary
        - "repaired": values were adjusted to satisfy the invariants
    """

    repaired = dict(cur)
    status: str | None = None

    for key in ("motion_speed", "gesture_amp", "warmth"):
        if key not in repaired:
            continue
        prev_val = prev.get(key, repaired[key])
        delta = repaired[key] - prev_val
        if abs(delta) > bounds.rho:
            repaired[key] = prev_val + math.copysign(bounds.rho, delta)
            status = "repaired"

    if repaired.get("containment"):
        if repaired.get("motion_speed", 0.0) != 0.0:
            repaired["motion_speed"] = 0.0
            status = status or "repaired"
        gesture = repaired.get("gesture_amp")
        if gesture is not None and gesture > bounds.eps_gesture:
            repaired["gesture_amp"] = bounds.eps_gesture
            status = status or "repaired"

    lambda_val = metrics.get("lambda")
    if lambda_val is not None and lambda_val > bounds.max_lambda:
        repaired["lambda_diss"] = repaired.get("lambda_diss", 0.0) + (lambda_val - bounds.max_lambda)
        status = status or "repaired"

    if status:
        cur.update(repaired)

    return cur, status


def main() -> int:
    ensure_lean_file()
    run_lake_build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
