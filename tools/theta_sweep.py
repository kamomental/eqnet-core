from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Tuple

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def extract_rows(path: Path) -> List[Tuple[float, float]]:
    rows: List[Tuple[float, float]] = []
    for event in iter_jsonl(path):
        receipt = ((event.get("meta") or {}).get("receipt") or {})
        qualia = receipt.get("qualia") or {}
        if not isinstance(qualia, dict):
            continue
        u = float(qualia.get("u_t", 0.0) or 0.0)
        load = float(qualia.get("load", 0.0) or 0.0)
        rows.append((u, load))
    return rows

def simulate(rows: List[Tuple[float, float]], *, k_u: float, k_l: float, u0: float, l0: float, theta: float, thr: float) -> float:
    if not rows:
        return 0.0
    allow = 0
    for u, load in rows:
        logit = k_u * (u0 - u) + k_l * (l0 - load) - theta
        p = sigmoid(logit)
        if p >= thr:
            allow += 1
    return allow / len(rows)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--k_u", type=float, default=2.0)
    ap.add_argument("--k_l", type=float, default=2.0)
    ap.add_argument("--u0", type=float, default=0.5)
    ap.add_argument("--l0", type=float, default=0.5)
    ap.add_argument("--theta-from", type=float, default=-3.0)
    ap.add_argument("--theta-to", type=float, default=3.0)
    ap.add_argument("--theta-step", type=float, default=0.25)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--target-lo", type=float, default=0.3)
    ap.add_argument("--target-hi", type=float, default=0.7)
    args = ap.parse_args()

    rows = extract_rows(Path(args.events))
    print(f"rows = {len(rows)}")

    theta = args.theta_from
    best = None
    while theta <= args.theta_to + 1e-9:
        rate = simulate(rows, k_u=args.k_u, k_l=args.k_l, u0=args.u0, l0=args.l0, theta=theta, thr=args.thr)
        mark = ""
        if args.target_lo <= rate <= args.target_hi and best is None:
            mark = "  <== target"
            best = theta
        print(f"theta={theta:+.2f}  open_rate?{rate:.3f}{mark}")
        theta += args.theta_step

    if best is not None:
        print(f"\nSuggested theta: {best:+.2f}")
    else:
        print("\nNo theta hit the target range; adjust sweep parameters.")

if __name__ == "__main__":
    main()
