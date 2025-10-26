#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified QUBO selector for candidates.v1 JSONL.

Auto-detects type per candidate (rag/etl/graph) and constructs diversity matrix D.

Modes:
- topk: select up to k items (soft via budget/cost), output JSON list of ids
- budget: select under budget (μ penalty), output JSON list of ids

Usage examples:
  python scripts/qubo_select.py --candidates candidates.rag.jsonl \
    --mode topk --k 10 --lam 0.25 --mu 0.05 --budget 8 --steps 60000

  python scripts/qubo_select.py --candidates candidates.etl.jsonl \
    --mode budget --budget 50 --lam 0.2 --mu 0.1 --steps 80000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from emot_terrain_lab.ops.qubo import build_qubo, solve_sa, SASchedule, cosine_similarity_matrix, autoschedule_from_Q


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i == 0 and 'candidates.v1' in line:
                # header line
                continue
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=str, required=True)
    ap.add_argument("--mode", type=str, choices=["topk", "budget"], required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--budget", type=float, default=10.0)
    ap.add_argument("--lam", type=float, default=0.25)
    ap.add_argument("--mu", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--redund_weight", type=float, default=0.2, help="weight of redundancy in ETL D matrix")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--auto_temp", action="store_true", help="auto temperature schedule from Q stats")
    ap.add_argument("--early_stop", action="store_true")
    ap.add_argument("--patience", type=int, default=10000)
    ap.add_argument("--out", type=str, default="", help="write provenance JSON to this file (else stdout)")
    ap.add_argument("--audit", type=str, default="", help="append a JSONL audit line to this path if set")
    args = ap.parse_args()

    rows = read_jsonl(Path(args.candidates))
    if not rows:
        print("[]")
        return

    ids = [r.get("id") for r in rows]
    gains = np.array([float(r.get("gain", 0.0)) for r in rows], dtype=np.float64)
    costs = np.array([float(r.get("cost", 1.0)) for r in rows], dtype=np.float64)
    embs = [np.asarray(r.get("emb", []), dtype=np.float64) for r in rows]
    # Decide type majority
    types = [r.get("type", "rag") for r in rows]
    type_set = set(types)
    # Diversity matrix
    if all(e.size > 0 for e in embs):
        D = cosine_similarity_matrix(np.stack(embs, axis=0))
    else:
        D = np.eye(len(rows), dtype=np.float64)
    # If ETL, blend redundancy scalar into D
    if type_set == {"etl"} or ("etl" in type_set and len(type_set) == 1):
        redund = np.array([float(r.get("redund", 0.0)) for r in rows], dtype=np.float64)
        D = (1.0 - args.redund_weight) * D + args.redund_weight * np.outer(redund, redund)

    Q = build_qubo(gains, D, cost=costs, budget=args.budget, lam=args.lam, mu=args.mu)
    sched = autoschedule_from_Q(Q, steps=args.steps) if args.auto_temp else SASchedule(steps=args.steps)
    x, E = solve_sa(Q, schedule=sched, seed=args.seed, patience=(args.patience if args.early_stop else None))
    sel = np.where(x > 0)[0].tolist()
    if args.mode == "topk" and len(sel) > args.k:
        sel = sorted(sel, key=lambda i: gains[i], reverse=True)[: args.k]
    sel_ids = [ids[i] for i in sorted(sel, key=lambda i: gains[i], reverse=True)]
    import time
    payload = {
        "schema": "candidates.v1",
        "version": "1.0",
        "seed": int(args.seed),
        "lam": float(args.lam),
        "mu": float(args.mu),
        "budget": float(args.budget),
        "steps": int(args.steps),
        "redund_weight": float(args.redund_weight),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": args.mode,
        "k": int(args.k),
        "selected": sel_ids,
    }
    out_path = args.out.strip()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(out_path).open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(payload, ensure_ascii=False))
    if args.audit:
        apath = Path(args.audit)
        apath.parent.mkdir(parents=True, exist_ok=True)
        with apath.open("a", encoding="utf-8") as af:
            af.write(json.dumps({"ts": payload["timestamp"], "mode": args.mode, "lam": args.lam, "mu": args.mu, "budget": args.budget, "steps": args.steps, "seed": args.seed, "n_candidates": len(rows), "n_selected": len(sel_ids) }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
