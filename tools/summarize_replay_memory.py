# -*- coding: utf-8 -*-
"""
Summarise replay memory traces for quick KPI checks.

Reads ``state/replay_memory.jsonl`` and prints aggregate metrics.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    mem_path = Path("state/replay_memory.jsonl")
    if not mem_path.exists():
        print("no memory")
        raise SystemExit(0)

    rows = [
        json.loads(line)
        for line in mem_path.open(encoding="utf-8")
        if line.strip()
    ]
    by_episode: dict[str, list[dict[str, object]]] = defaultdict(list)
    value_hist: list[float] = []
    for row in rows:
        episode = str(row.get("episode_id", ""))
        by_episode[episode].append(row)
        value = float(row.get("value", {}).get("success", 0.5))
        consistency = float(row.get("value", {}).get("consistency", 0.0))
        effort = float(row.get("value", {}).get("effort", 0.0))
        value_hist.append(value - consistency - 0.2 * effort)

    freq = np.mean([len(v) for v in by_episode.values()]) if by_episode else 0.0
    print("replay_freq_per_episode", float(freq))
    if value_hist:
        print("V_mean", float(np.mean(value_hist)), "V_std", float(np.std(value_hist)))
    else:
        print("V_mean", 0.0, "V_std", 0.0)
    # Placeholder: hook actual calibration logic when live measurements land.


if __name__ == "__main__":
    main()
