#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Select ETL items via QUBO (Δaff impact vs redundancy vs risk/cost under budget).

Input JSON (list):
[
  {"id":"ep1","impact":0.9,"redundancy":0.2,"risk":0.1,"cost":1.0},
  ...
]

Formulation (example):
  gain = impact - beta*redundancy - gamma*risk
  loss = -gain@x + lam * x^T D x + mu * (cost@x - budget)^2
Here we approximate D by outer(redundancy, redundancy) to discourage co-selection
of redundant items. For more precise modeling, provide a pairwise redundancy matrix.
"""

from __future__ import annotations

import argparse
import json
from typing import List

import numpy as np

from emot_terrain_lab.ops.qubo import build_qubo, solve_sa, SASchedule


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=str, required=True)
    ap.add_argument("--budget", type=float, default=50.0)
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--mu", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.2, help="redundancy weight in gain")
    ap.add_argument("--gamma", type=float, default=0.1, help="risk weight in gain")
    ap.add_argument("--steps", type=int, default=80000)
    args = ap.parse_args()

    items = json.load(open(args.candidates, "r", encoding="utf-8"))
    ids = [it["id"] for it in items]
    impact = np.array([float(it.get("impact", 0.0)) for it in items], dtype=np.float64)
    redund = np.array([float(it.get("redundancy", 0.0)) for it in items], dtype=np.float64)
    risk = np.array([float(it.get("risk", 0.0)) for it in items], dtype=np.float64)
    cost = np.array([float(it.get("cost", 1.0)) for it in items], dtype=np.float64)

    gain = impact - args.beta * redund - args.gamma * risk
    D = np.outer(redund, redund)  # simple redundancy coupling
    Q = build_qubo(gain, D, cost=cost, budget=args.budget, lam=args.lam, mu=args.mu)
    x, E = solve_sa(Q, schedule=SASchedule(steps=args.steps))
    for i, keep in enumerate(x):
        if int(keep) == 1:
            print(ids[i])


if __name__ == "__main__":
    main()

