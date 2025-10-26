# -*- coding: utf-8 -*-
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from terrain.system import EmotionalMemorySystem

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, required=True)
    ap.add_argument("--state", type=str, default="data/state")
    ap.add_argument("--user", type=str, default="user_000")
    args = ap.parse_args()
    Path(args.state).mkdir(parents=True, exist_ok=True)
    system = EmotionalMemorySystem(args.state)

    with open(args.inp, "r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("user_id") != args.user:
                continue
            system.ingest_dialogue(
                row["user_id"],
                row["dialogue"],
                row["timestamp"],
            )

    system.daily_consolidation()

if __name__ == "__main__":
    main()
