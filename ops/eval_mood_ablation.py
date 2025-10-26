#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A/B evaluation scaffold for mood gate (m-ON/OFF).

Runs two conditions for N trials each:
- OFF: no mood_* in metrics
- ON: inject mood_* via EQNET_MOOD_METRICS env

Writes JSONL with controls and metrics per trial to reports/.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Dict, Any

import numpy as np

from emot_terrain_lab.hub.runtime import EmotionalHubRuntime


def _trial_once(cond: str, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    # Inject or clear mood metrics via env
    if cond == "on":
        mood = {
            "mood_v": float(rng.uniform(-0.2, 0.5)),
            "mood_a": float(rng.uniform(-0.1, 0.6)),
            "mood_effort": float(rng.uniform(0.0, 0.4)),
            "mood_uncertainty": float(rng.uniform(0.0, 0.5)),
        }
        os.environ["EQNET_MOOD_METRICS"] = json.dumps(mood)
    else:
        os.environ.pop("EQNET_MOOD_METRICS", None)

    rt = EmotionalHubRuntime()
    out = rt.step(user_text=None)  # controls/metrics without LLM call
    controls = rt.last_controls
    metrics = rt.last_metrics
    rec: Dict[str, Any] = {
        "condition": cond,
        "controls": {
            "temperature": float(controls.temperature) if controls else None,
            "top_p": float(controls.top_p) if controls else None,
            "pause_ms": float(controls.pause_ms) if controls else None,
        },
        "metrics": {k: float(v) for k, v in (metrics or {}).items()},
    }
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30, help="Trials per condition (ON/OFF)")
    ap.add_argument(
        "--out",
        type=str,
        default=os.path.join("reports", "mood_ablation.jsonl"),
        help="Output JSONL path",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with open(args.out, "w", encoding="utf-8") as f:
        for cond in ("off", "on"):
            for i in range(args.trials):
                rec = _trial_once(cond, int(rng.integers(0, 1_000_000)))
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[mood-ablation] wrote {args.trials*2} records to {args.out}")


if __name__ == "__main__":
    main()

