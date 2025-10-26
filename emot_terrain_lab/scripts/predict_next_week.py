# -*- coding: utf-8 -*-
"""
直近ログをリプレイしながら、地形ベースの簡易予測性能を算出するスクリプト。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from terrain.emotion import AXES
from terrain.system import EmotionalMemorySystem


def predict_from_terrain(system: EmotionalMemorySystem, fallback: np.ndarray) -> np.ndarray:
    """最新のエピソード中心ベクトルから次週の感情を予測。なければフォールバック。"""
    if not system.l2.episodes:
        return fallback
    centers = [
        np.array(ep["emotion_pattern"]["center"], dtype=float)
        for ep in system.l2.episodes[-5:]
    ]
    return np.mean(centers, axis=0)


def load_logs(path: Path, user_id: str) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    rows = [row for row in rows if row.get("user_id") == user_id]
    rows.sort(key=lambda row: row["timestamp"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=str, default="data/state")
    parser.add_argument("--in", dest="inp", type=str, required=True)
    parser.add_argument("--out", type=str, default="data/preds.csv")
    parser.add_argument("--user", type=str, default="user_000")
    args = parser.parse_args()

    input_path = Path(args.inp)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    system = EmotionalMemorySystem(args.state)
    logs = load_logs(input_path, args.user)

    previous_vec = np.zeros(len(AXES), dtype=float)
    rows: list[dict[str, object]] = []

    for record in logs:
        prediction = predict_from_terrain(system, previous_vec)
        truth = np.array(record["emotion_vec"], dtype=float)
        mae = float(np.mean(np.abs(prediction - truth)))
        rows.append({"timestamp": record["timestamp"], "mae": mae})

        system.ingest_dialogue(record["user_id"], record["dialogue"], record["timestamp"])
        system.daily_consolidation()
        previous_vec = truth

    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestamp", "mae"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
