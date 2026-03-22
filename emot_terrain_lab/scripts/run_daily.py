# -*- coding: utf-8 -*-
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from terrain.system import EmotionalMemorySystem
from memory.inner_os_working_memory_bridge import write_inner_os_working_memory_snapshot
from sleep.inner_os_bridge import write_inner_os_sleep_snapshot_for_system

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, required=True)
    ap.add_argument("--state", type=str, default="data/state")
    ap.add_argument("--user", type=str, default="user_000")
    ap.add_argument(
        "--inner-os-sleep-out",
        type=str,
        default=None,
        help="Write a reusable inner_os sleep-consolidation snapshot after daily consolidation",
    )
    ap.add_argument(
        "--inner-os-working-memory-out",
        type=str,
        default=None,
        help="Write a reusable inner_os working-memory snapshot after daily consolidation",
    )
    ap.add_argument(
        "--inner-os-memory-path",
        type=str,
        default=None,
        help="Optional source JSONL path for inner_os working-memory traces",
    )
    args = ap.parse_args()
    state_dir = Path(args.state)
    state_dir.mkdir(parents=True, exist_ok=True)
    system = EmotionalMemorySystem(str(state_dir))

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
    sleep_out = args.inner_os_sleep_out or str(state_dir / "inner_os_sleep_snapshot.json")
    write_inner_os_sleep_snapshot_for_system(system, out_path=sleep_out)
    working_memory_out = args.inner_os_working_memory_out or str(state_dir / "inner_os_working_memory_snapshot.json")
    write_inner_os_working_memory_snapshot(
        out_path=working_memory_out,
        memory_path=args.inner_os_memory_path,
    )

if __name__ == "__main__":
    main()
