from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

_THIN_TOP_LEVEL_KEYS = {
    "turn_id",
    "timestamp_ms",
    "timestamp",
    "seed",
    "scenario_id",
    "session_id",
    "tags",
    "user_text",
    "somatic",
    "context",
    "world",
    "mood",
    "metrics",
    "gate_context",
    "culture",
    "emotion",
    "trace_observations",
    "observations",
    "runtime_observations",
    "talk_mode",
    "emotion_tag",
}


def _as_mapping(obj: Any) -> Mapping[str, Any] | None:
    """Best-effort conversion to a mapping without exploding payload size."""

    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj
    if is_dataclass(obj):
        try:
            return asdict(obj)
        except Exception:
            return None
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump()
        except Exception:
            return None
    dict_fn = getattr(obj, "dict", None)
    if callable(dict_fn):
        try:
            return dict_fn()
        except Exception:
            return None
    if hasattr(obj, "__dict__"):
        try:
            return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
        except Exception:
            return None
    return None


def _coerce_nested(value: Any) -> Any:
    mapping = _as_mapping(value)
    if mapping is not None:
        return dict(mapping)
    return value


def _ensure_timestamp_ms(payload: dict[str, Any], source: Any) -> None:
    if "timestamp_ms" in payload:
        return
    stamp = payload.get("timestamp")
    if stamp is None and hasattr(source, "timestamp"):
        stamp = getattr(source, "timestamp")
    if isinstance(stamp, datetime):
        payload["timestamp_ms"] = int(stamp.timestamp() * 1000)
    elif isinstance(stamp, (int, float)):
        payload["timestamp_ms"] = int(float(stamp))


def to_moment_entry(obj: Any) -> dict[str, Any]:
    """Return a thin mapping describing ``obj`` for mapper/trace consumption."""

    raw = _as_mapping(obj)
    entry: dict[str, Any] = {}
    if raw is not None:
        for key in _THIN_TOP_LEVEL_KEYS:
            if key in raw:
                entry[key] = _coerce_nested(raw[key])
    else:
        for key in _THIN_TOP_LEVEL_KEYS:
            if hasattr(obj, key):
                entry[key] = _coerce_nested(getattr(obj, key))

    _ensure_timestamp_ms(entry, obj)

    extras = entry.get("extras")
    if not isinstance(extras, dict):
        extras = {}
        entry["extras"] = extras
    extras.setdefault("raw_ref", {"type": type(obj).__name__})

    return entry


__all__ = ["to_moment_entry"]

