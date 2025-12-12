#!/usr/bin/env python3
"""Lightweight guardrails for the ForceMatrix logic."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from eqnet_core.models.conscious import ForceMatrix, LayerForceRow, SelfLayer


def _assert_close(actual: float, expected: float, *, tol: float = 1e-9) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"expected {expected}, got {actual}")


def main() -> None:
    base = ForceMatrix(
        reflex=LayerForceRow(survival=0.8, physiological=0.6),
        affective=LayerForceRow(social=0.9, attachment=0.7),
        narrative=LayerForceRow(exploration=0.9),
    )
    overrides = {
        "reflex": {"survival": 0.95},
        "narrative": {"social": 0.4},
    }
    merged = base.merge(overrides)

    _assert_close(merged.row_for(SelfLayer.REFLEX).survival, 0.95)
    _assert_close(merged.row_for(SelfLayer.REFLEX).physiological, 0.6)
    _assert_close(merged.row_for(SelfLayer.NARRATIVE).social, 0.4)
    _assert_close(merged.row_for(SelfLayer.NARRATIVE).attachment, 0.0)

    fallback = ForceMatrix.from_mapping(None)
    _assert_close(fallback.row_for(SelfLayer.AFFECTIVE).survival, 0.0)

    print("OK: ForceMatrix merge/query smoke passed")


if __name__ == "__main__":
    main()
