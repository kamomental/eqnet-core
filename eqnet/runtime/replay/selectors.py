from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping


def day_key_from_record(record: Mapping[str, Any]) -> str | None:
    ts = record.get("timestamp_ms")
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(float(ts) / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
        except Exception:
            return None
    return None


def group_by_day(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        day_key = day_key_from_record(record)
        if not day_key:
            continue
        grouped.setdefault(day_key, []).append(record)
    return grouped


def select_day_range(
    grouped: Dict[str, List[Dict[str, Any]]],
    *,
    start_day_key: str | None,
    end_day_key: str | None,
) -> Dict[str, List[Dict[str, Any]]]:
    keys = sorted(grouped.keys())
    selected: Dict[str, List[Dict[str, Any]]] = {}
    for key in keys:
        if start_day_key and key < start_day_key:
            continue
        if end_day_key and key > end_day_key:
            continue
        selected[key] = grouped[key]
    return selected

