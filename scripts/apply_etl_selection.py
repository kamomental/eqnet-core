#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Idempotent apply of ETL selection with provenance.

Reads selected.etl.json (from qubo_select.py --out) and emits apply_log.jsonl with actions:
  upsert | skip | deprecate

This is a placeholder that records intent; integrate with your actual ETL upgrader.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_selected(path: Path) -> dict:
    return json.load(path.open("r", encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", type=str, required=True)
    ap.add_argument("--apply_log", type=str, default="out/apply_etl_log.jsonl")
    ap.add_argument("--gen", type=str, default="auto", help="generation id (YYYYMMDD or 'auto')")
    ap.add_argument("--keep_last_n", type=int, default=5)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    sel = read_selected(Path(args.selected))
    ids = list(sel.get("selected", []))
    gen = sel.get("timestamp", "na").replace("-", "").split("T")[0] if args.gen == "auto" else args.gen
    Path(args.apply_log).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.apply_log).open("a", encoding="utf-8") as log:
        for _id in ids:
            rec = {"action": "upsert", "id": _id, "gen": gen, "dry": bool(args.dry_run)}
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Applied {len(ids)} ETL items (gen={gen}) [dry={args.dry_run}] → {args.apply_log}")


if __name__ == "__main__":
    main()

