#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RAG re-ranking via QUBO (accuracy vs diversity vs budget).

Input JSON (list):
[
  {"id":"doc1","gain":1.2,"cost":1.0,"emb":[...]},
  {"id":"doc2","gain":0.8,"cost":1.0,"emb":[...]},
  ...
]

Usage:
  python scripts/qubo_rerank_rag.py --candidates candidates.json \
    --k 10 --lam 0.25 --mu 0.05 --budget 8 --steps 60000

Outputs selected IDs (one per line) ordered by gain within selected set.
"""

from __future__ import annotations

import argparse
import json
from typing import List

import numpy as np

from emot_terrain_lab.ops.qubo import build_qubo, solve_sa, SASchedule, cosine_similarity_matrix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=str, required=True)
    ap.add_argument("--k", type=int, default=10, help="target selection size (soft via budget)")
    ap.add_argument("--lam", type=float, default=0.25, help="diversity weight")
    ap.add_argument("--mu", type=float, default=0.05, help="budget penalty")
    ap.add_argument("--budget", type=float, default=8.0, help="sum(cost) target")
    ap.add_argument("--steps", type=int, default=60000)
    args = ap.parse_args()

    items = json.load(open(args.candidates, "r", encoding="utf-8"))
    ids = [it["id"] for it in items]
    gain = np.array([float(it.get("gain", 0.0)) for it in items], dtype=np.float64)
    cost = np.array([float(it.get("cost", 1.0)) for it in items], dtype=np.float64)
    embs = [np.asarray(it.get("emb", []), dtype=np.float64) for it in items]
    if any(e.size == 0 for e in embs):
        D = np.eye(len(items), dtype=np.float64)
    else:
        D = cosine_similarity_matrix(np.stack(embs, axis=0))
    # Penalize similarity (encourage diversity)
    Q = build_qubo(gain, D, cost=cost, budget=args.budget, lam=args.lam, mu=args.mu)
    x, E = solve_sa(Q, schedule=SASchedule(steps=args.steps))
    sel = np.where(x > 0)[0].tolist()
    # Keep top-k if too many selected
    if len(sel) > args.k:
        sel = sorted(sel, key=lambda i: gain[i], reverse=True)[: args.k]
    # Order by gain (or any secondary rule)
    sel_sorted = sorted(sel, key=lambda i: gain[i], reverse=True)
    for i in sel_sorted:
        print(ids[i])


if __name__ == "__main__":
    main()

