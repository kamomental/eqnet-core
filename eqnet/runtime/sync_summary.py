from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping
import json


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, Mapping):
                items.append(dict(row))
    return items


def summarize_sync_realtime(telemetry_dir: Path, day: date) -> Dict[str, Any]:
    stamp = day.strftime("%Y%m%d")
    micro_rows: List[Dict[str, Any]] = []
    for path in sorted(telemetry_dir.glob(f"sync_micro_outcomes-{stamp}.jsonl")):
        micro_rows.extend(_read_jsonl(path))
    downshift_rows: List[Dict[str, Any]] = []
    for path in sorted(telemetry_dir.glob(f"sync_downshifts-{stamp}.jsonl")):
        downshift_rows.extend(_read_jsonl(path))

    counts = {"HELPED": 0, "HARMED": 0, "UNKNOWN": 0, "NO_EFFECT": 0}
    for row in micro_rows:
        result = str(row.get("result") or "").upper()
        if result in counts:
            counts[result] += 1
    total = sum(counts.values())
    harmed_rate = (float(counts["HARMED"]) / float(total)) if total > 0 else 0.0
    unknown_rate = (float(counts["UNKNOWN"]) / float(total)) if total > 0 else 0.0
    helped_rate = (float(counts["HELPED"]) / float(total)) if total > 0 else 0.0

    day_start_ms = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)
    day_end_ms = day_start_ms + 86_400_000 - 1
    intervals: List[tuple[int, int]] = []
    for row in downshift_rows:
        start = row.get("timestamp_ms")
        end = row.get("cooldown_until_ts_ms")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        s = max(day_start_ms, int(start))
        e = min(day_end_ms, int(end))
        if e > s:
            intervals.append((s, e))
    intervals.sort(key=lambda it: (it[0], it[1]))
    merged: List[tuple[int, int]] = []
    for s, e in intervals:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            prev_s, prev_e = merged[-1]
            merged[-1] = (prev_s, max(prev_e, e))
    suppressed_ms = sum(max(0, e - s) for s, e in merged)
    suppressed_ratio = (float(suppressed_ms) / 86_400_000.0) if suppressed_ms > 0 else 0.0

    return {
        "sync_micro_helped_count": int(counts["HELPED"]),
        "sync_micro_harmed_count": int(counts["HARMED"]),
        "sync_micro_unknown_count": int(counts["UNKNOWN"]),
        "sync_micro_no_effect_count": int(counts["NO_EFFECT"]),
        "sync_micro_helped_rate": round(helped_rate, 6),
        "sync_micro_harmed_rate": round(harmed_rate, 6),
        "sync_micro_unknown_rate": round(unknown_rate, 6),
        "sync_downshift_applied_count": int(len(downshift_rows)),
        "sync_emit_suppressed_time_ratio": round(suppressed_ratio, 6),
    }
