#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Idempotent apply of RAG selection with provenance.

Reads selected.rag.json (from qubo_select.py --out) and emits apply_log.jsonl with actions:
  upsert | skip | deprecate

This is a placeholder that records intent; integrate with your actual indexer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Set


def read_selected(path: Path) -> dict:
    return json.load(path.open("r", encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", type=str, required=True)
    ap.add_argument("--apply_log", type=str, default="out/apply_rag_log.jsonl")
    ap.add_argument("--gen", type=str, default="auto", help="generation id (YYYYMMDD or 'auto')")
    ap.add_argument("--keep_last_n", type=int, default=5)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--warm_cache", type=str, default="out/warm_cache.json", help="write warm-cache prior file for retriever")
    ap.add_argument("--prior_epsilon", type=float, default=0.05, help="prior bias added to selected IDs in retriever")
    args = ap.parse_args()

    sel = read_selected(Path(args.selected))
    ids = list(sel.get("selected", []))
    gen = sel.get("timestamp", "na").replace("-", "").split("T")[0] if args.gen == "auto" else args.gen
    Path(args.apply_log).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.apply_log).open("a", encoding="utf-8") as log:
        for _id in ids:
            rec = {"action": "upsert", "id": _id, "gen": gen, "dry": bool(args.dry_run)}
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # Warm cache prior
    try:
        wc_path = Path(args.warm_cache)
        wc_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ids": ids, "epsilon": float(args.prior_epsilon), "timestamp": sel.get("timestamp", "na")}
        wc_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Warm cache written: {wc_path} (ids={len(ids)}, eps={args.prior_epsilon})")
        print("Hint: set EQNET_RAG_WARM_CACHE=", str(wc_path))
    except Exception as e:
        print("[warn] warm cache not written:", e)
    print(f"Applied {len(ids)} RAG items (gen={gen}) [dry={args.dry_run}] → {args.apply_log}")


if __name__ == "__main__":
    main()
