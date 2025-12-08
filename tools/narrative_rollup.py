#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Roll up self_report logs into coarse narrative summaries."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def chunk_entries(entries: List[Dict[str, Any]], window: int) -> Iterable[List[Dict[str, Any]]]:
    for idx in range(0, len(entries), window):
        chunk = entries[idx : idx + window]
        if chunk:
            yield chunk


def mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def infer_theme(avg_love: float, avg_trust: float, avg_override: float) -> str:
    if avg_override >= 0.3:
        return "fastpath_exploration"
    if avg_trust < 0.45:
        return "guarded"
    if avg_love >= 0.22:
        return "warmth"
    if avg_love < 0.12:
        return "cautious"
    return "steady"


def summarize_chunk(chunk: List[Dict[str, Any]]) -> Dict[str, Any]:
    episodes = [entry.get("episode_id") for entry in chunk if entry.get("episode_id")]
    tags = {}
    love_values: List[float] = []
    trust_values: List[float] = []
    override_values: List[float] = []
    for entry in chunk:
        tag = entry.get("tag") or "default"
        tags[tag] = tags.get(tag, 0) + 1
        metrics = entry.get("metrics") or {}
        love_values.append(float(metrics.get("love", 0.0) or 0.0))
        trust_values.append(float(metrics.get("intent_trust", 0.0) or 0.0))
        override_values.append(float(metrics.get("fastpath_override", 0.0) or 0.0))

    avg_love = mean(love_values)
    avg_trust = mean(trust_values)
    avg_override = mean(override_values)
    theme = infer_theme(avg_love, avg_trust, avg_override)
    start_ts = chunk[0].get("ts")
    end_ts = chunk[-1].get("ts")
    period = {
        "start": dt.datetime.utcfromtimestamp(start_ts).isoformat() + "Z" if start_ts else None,
        "end": dt.datetime.utcfromtimestamp(end_ts).isoformat() + "Z" if end_ts else None,
    }
    description = (
        f"avg love={avg_love:.2f}, trust={avg_trust:.2f}, override={avg_override:.2f}. "
        f"theme={theme} across {len(chunk)} episodes."
    )
    return {
        "narrative_id": f"{episodes[0]}_{episodes[-1]}" if episodes else f"chunk_{start_ts}",
        "episode_ids": episodes,
        "tag_counts": tags,
        "theme": theme,
        "emotional_trend": "warm" if avg_love >= 0.22 else "steady",
        "risk_trend": "low" if avg_trust >= 0.6 else "guarded",
        "description": description,
        "period": period,
        "metrics": {
            "avg_love": avg_love,
            "avg_trust": avg_trust,
            "avg_override": avg_override,
        },
        "monuments": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-report", type=Path, default=Path("logs/self_report.jsonl"))
    parser.add_argument("--window", type=int, default=8, help="episodes per narrative chunk")
    parser.add_argument("--out", type=Path, default=Path("logs/narrative_log.jsonl"))
    args = parser.parse_args()

    entries = read_jsonl(args.self_report)
    if not entries:
        print(f"[narrative_rollup] self-report log not found at {args.self_report}")
        return

    narratives = [summarize_chunk(chunk) for chunk in chunk_entries(entries, args.window)]
    if not narratives:
        print("[narrative_rollup] no narrative chunks created")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in narratives:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[narrative_rollup] wrote {len(narratives)} entries to {args.out}")


if __name__ == "__main__":
    main()
