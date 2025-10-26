#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tail logs/alerts.jsonl and pretty-print alert events.

Usage:
  python scripts/alerts_tail.py --file logs/alerts.jsonl --follow
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def pretty(alert: dict) -> str:
    t = alert.get("type", "?")
    stage = alert.get("stage", "")
    step = alert.get("step", "")
    if t == "ignite":
        return f"ignite stage={stage} step={step} I={alert.get('I'):.3f} dR={alert.get('delta_R'):.3f} z={alert.get('entropy_z'):.3f} ms={alert.get('ignite_ms')}"
    if t == "inhibit":
        return f"inhibit stage={stage} step={step} ms={alert.get('ignite_ms')} action={alert.get('action')}"
    if t == "autonomy.downshift":
        return f"downshift stage={stage} step={step} trust={alert.get('intent_trust'):.2f}"
    return json.dumps(alert, ensure_ascii=False)


def tail(path: Path, follow: bool) -> None:
    if not path.exists():
        print(f"No alert file: {path}")
        return
    with path.open("r", encoding="utf-8") as fh:
        if follow:
            fh.seek(0, 2)
            while True:
                line = fh.readline()
                if not line:
                    time.sleep(0.3)
                    continue
                try:
                    alert = json.loads(line)
                    print(pretty(alert))
                except Exception:
                    print(line.rstrip())
        else:
            for line in fh:
                try:
                    alert = json.loads(line)
                    print(pretty(alert))
                except Exception:
                    print(line.rstrip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, default="logs/alerts.jsonl")
    ap.add_argument("--follow", action="store_true")
    args = ap.parse_args()
    tail(Path(args.file), args.follow)


if __name__ == "__main__":
    main()

