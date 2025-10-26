# -*- coding: utf-8 -*-
"""
Estimate Green-function kernel and resonance gain from decision logs.

usage:
    python tools/tune_green_kernel.py \
        --log logs/decisions.jsonl \
        --out reports/green_kernel_suggestion.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

AFFECT_AXES = ("v", "a", "d", "n", "c", "e", "s")


def load_examples(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        resp = rec.get("green", {}).get("response", {})
        delta = resp.get("delta")
        qualia = resp.get("qualia_vector")
        if not delta or not qualia:
            continue
        rows.append(
            {
                "qualia": [
                    float(qualia.get("sensation", 0.0)),
                    float(qualia.get("meaning", 0.0)),
                ],
                "delta": [float(delta.get(axis, 0.0)) for axis in AFFECT_AXES],
                "heartiness": float(rec.get("heartiness", 0.0)),
            }
        )
    return rows


def fit_kernel(rows: List[dict]) -> Dict[str, List[float]]:
    if not rows:
        return {axis: [0.0, 0.0] for axis in AFFECT_AXES}
    X = np.array([row["qualia"] for row in rows], dtype=np.float32)
    kernel = {}
    for idx, axis in enumerate(AFFECT_AXES):
        y = np.array([row["delta"][idx] for row in rows], dtype=np.float32)
        if not np.any(y):
            coeffs = np.zeros(2, dtype=np.float32)
        else:
            coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        kernel[axis] = coeffs.round(6).tolist()
    return kernel


def estimate_resonance(rows: List[dict]) -> float:
    if not rows:
        return 0.3
    magnitudes = [abs(row["delta"][0]) + abs(row["delta"][1]) for row in rows]
    heartiness = [row["heartiness"] for row in rows]
    if not magnitudes or not heartiness:
        return 0.3
    m = float(np.mean(magnitudes))
    h = float(np.mean(heartiness))
    if h < 1e-6:
        return 0.3
    return round(min(1.0, max(0.05, m / (h * 4.0))), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=Path("logs/decisions.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("reports/green_kernel_suggestion.json"))
    args = ap.parse_args()

    rows = load_examples(args.log)
    kernel = fit_kernel(rows)
    resonance = estimate_resonance(rows)
    summary = {
        "suggested_kernel": kernel,
        "suggested_culture_resonance": resonance,
        "samples": len(rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
