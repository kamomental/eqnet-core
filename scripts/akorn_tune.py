#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AKOrN tuning helper: preview control deltas for given metrics.

Lets you experiment with small-gain parameters and see how temperature/top_p/
pause_ms would shift across a small grid of metric conditions.

Usage examples:
  # preview with env overrides
  $env:AKORN_TEMP_GAIN_R = "-0.12"; python scripts/akorn_tune.py

  # override via args (takes precedence)
  python scripts/akorn_tune.py --temp_gain_R -0.12 --temp_gain_I -0.10 --top_p_gain_R -0.08 --top_p_gain_I -0.06 --pause_gain_R 200
"""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

from emot_terrain_lab.hub.akorn import AkornGate, AkornConfig


def apply_overrides(cfg: AkornConfig, args: argparse.Namespace) -> AkornConfig:
    d = vars(cfg)
    for k in list(d.keys()):
        if hasattr(args, k) and getattr(args, k) is not None:
            setattr(cfg, k, getattr(args, k))
    return cfg


def preview(cfg: AkornConfig) -> None:
    gate = AkornGate(cfg)
    base = {"temperature": 0.65, "top_p": 0.85, "pause_ms": 360.0}
    metrics_grid = [
        {"R": r, "rho": rho, "I": I, "q": q}
        for r in (0.4, 0.6, 0.8)
        for rho in (1.0, 1.2)
        for I in (-0.2, 0.0, 0.2)
        for q in (0.0, 0.5, 1.0)
    ]
    dtemp_min = dtemp_max = 0.0
    dtop_min = dtop_max = 0.0
    dpause_min = dpause_max = 0.0
    for m in metrics_grid:
        out, log = gate.apply(dict(base), m)
        dtemp = out["temperature"] - base["temperature"]
        dtop = out["top_p"] - base["top_p"]
        dpause = out["pause_ms"] - base["pause_ms"]
        dtemp_min = min(dtemp_min, dtemp); dtemp_max = max(dtemp_max, dtemp)
        dtop_min = min(dtop_min, dtop); dtop_max = max(dtop_max, dtop)
        dpause_min = min(dpause_min, dpause); dpause_max = max(dpause_max, dpause)
    print("AKOrN deltas across grid:")
    print(f"  temperature: {dtemp_min:+.3f} .. {dtemp_max:+.3f}")
    print(f"  top_p:       {dtop_min:+.3f} .. {dtop_max:+.3f}")
    print(f"  pause_ms:    {dpause_min:+.1f} .. {dpause_max:+.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    # small-gain knobs
    ap.add_argument("--temp_gain_R", type=float)
    ap.add_argument("--temp_gain_I", type=float)
    ap.add_argument("--top_p_gain_R", type=float)
    ap.add_argument("--top_p_gain_I", type=float)
    ap.add_argument("--pause_gain_R", type=float)
    ap.add_argument("--pause_gain_I", type=float)
    args = ap.parse_args()

    cfg = AkornConfig.from_env()
    cfg = apply_overrides(cfg, args)
    preview(cfg)


if __name__ == "__main__":
    main()

