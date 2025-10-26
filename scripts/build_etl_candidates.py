#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build QUBO candidates JSONL for ETL selection (candidates.v1).

Input: episodes JSONL, one per line, fields (best-effort):
  {"id": str (opt), "timestamp": str, "delta_affect": float or dict, "topic": str (opt),
   "emb": [float,...] (opt), "usage_count": int (opt), "pii_flag": bool (opt), "leak_score": float (opt)}

Output JSONL (candidates.v1):
  {"id":"ep:<timestamp>","type":"etl","gain":<float>,"cost":1.0,
   "redund":<float>,"risk":<float>,"emb":[...],"meta":{...}}

Scoring (baseline):
  impact = |Δaff| * log(1 + usage_count)
  redund = topic内の簡易密度（近傍が多いほど↑） or provided
  risk = 1.0 if pii_flag else 0.0 + leak_score
  gain = impact - beta*redund - gamma*risk
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List
import json
import math

import numpy as np


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def get_delta_aff(ep: Dict[str, Any]) -> float:
    v = ep.get("delta_affect")
    if isinstance(v, dict):
        return float(v.get("valence", 0.0))
    try:
        return float(v)
    except Exception:
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--r_pii", type=float, default=1.0, help="risk weight for PII flag")
    ap.add_argument("--r_leak", type=float, default=0.5, help="risk weight for leak score")
    ap.add_argument("--r_policy", type=float, default=0.5, help="risk weight for policy violation score")
    args = ap.parse_args()

    eps = read_jsonl(Path(args.episodes))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Topic-wise simple density (count per topic)
    topic_counts: Dict[str, int] = {}
    for ep in eps:
        t = str(ep.get("topic", ""))
        topic_counts[t] = topic_counts.get(t, 0) + 1

    with out.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"schema": "candidates.v1", "version": "1.0"}, ensure_ascii=False) + "\n")
        for ep in eps:
            ts = ep.get("timestamp") or ep.get("id") or "na"
            _id = f"ep:{ts}"
            da = abs(get_delta_aff(ep))
            usage = float(ep.get("usage_count", 0.0))
            impact = da * math.log1p(usage)
            topic = str(ep.get("topic", ""))
            redund = float(topic_counts.get(topic, 1) - 1) / max(1.0, float(len(eps)))
            pii = 1.0 if ep.get("pii_flag") else 0.0
            leak = float(ep.get("leak_score", 0.0))
            pviol = float(ep.get("policy_violation_score", 0.0))
            risk = args.r_pii * pii + args.r_leak * leak + args.r_policy * pviol
            gain = impact - args.beta * redund - args.gamma * risk
            record = {
                "id": _id,
                "type": "etl",
                "gain": float(gain),
                "cost": 1.0,
                "redund": float(redund),
                "risk": float(risk),
                "emb": ep.get("emb", []),
                "meta": {"Δaff": da, "topic": topic, "usage": usage, "pii": int(pii), "leak_score": leak, "policy_violation_score": pviol},
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
