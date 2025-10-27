"""Lightweight event logger used across runtime + nightly tooling."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

DEFAULT_LOG = Path(os.getenv("TELEMETRY_EVENT_LOG", "logs/telemetry_events.jsonl"))


def event(name: str, payload: Mapping[str, Any], *, log_path: str | Path | None = None) -> dict:
    """Record an event with a timestamp and persist to JSONL."""
    log_file = Path(log_path) if log_path else DEFAULT_LOG
    log_file.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": time.time(), "event": str(name), "data": _coerce_payload(payload)}
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def _coerce_payload(payload: Mapping[str, Any]) -> dict:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        out[key] = _coerce_value(value)
    return out


def _coerce_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {k: _coerce_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_coerce_value(v) for v in value]
    try:
        return float(value)
    except Exception:
        return str(value)


__all__ = ["event"]
