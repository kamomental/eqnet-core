from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import date, datetime, timezone
from typing import Any, Optional


def resolve_day_key_from_moment(timestamp: datetime) -> str:
    return timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d")


def resolve_day_key_from_date(day: date) -> str:
    return day.strftime("%Y-%m-%d")


def resolve_day_key_from_as_of(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return resolve_day_key_from_moment(value)
    if isinstance(value, date):
        return resolve_day_key_from_date(value)
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def get_or_create_episode_id(entry: Any) -> str:
    explicit = _read_entry(entry, "episode_id")
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()

    seed_payload = {
        "session_id": _read_entry(entry, "session_id"),
        "scenario_id": _read_entry(entry, "scenario_id"),
        "turn_id": _read_entry(entry, "turn_id"),
        "timestamp_ms": _safe_timestamp_ms(_read_entry(entry, "timestamp")),
    }
    raw = json.dumps(seed_payload, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"ep-{digest}"


def _read_entry(entry: Any, key: str) -> Any:
    if isinstance(entry, Mapping):
        return entry.get(key)
    return getattr(entry, key, None)


def _safe_timestamp_ms(value: Any) -> Optional[int]:
    if isinstance(value, datetime):
        return int(value.astimezone(timezone.utc).timestamp() * 1000)
    return None

