from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from eqnet.contracts import TurnResult

META_FIELDS = (
    "schema_version",
    "source_loop",
    "runtime_version",
    "idempotency_key",
    "scenario_id",
    "turn_id",
    "seed",
    "timestamp_ms",
)


def _default_serializer(obj: Any) -> Any:
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def _inject_meta(trace_dict: dict[str, Any], meta: Mapping[str, Any]) -> dict[str, Any]:
    for key in META_FIELDS:
        if key not in trace_dict or trace_dict[key] is None:
            trace_dict[key] = meta.get(key)
    return trace_dict


def write_trace_jsonl(path: Path | str, result: TurnResult, meta: Mapping[str, Any]) -> None:
    """Append ``result.trace`` to ``path`` as one JSON line."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    trace_dict = result.trace.to_dict()
    _inject_meta(trace_dict, meta)
    with target.open("a", encoding="utf-8") as handle:
        json.dump(trace_dict, handle, ensure_ascii=False, default=_default_serializer)
        handle.write("\n")


def append_trace_event(path: Path | str, record: Mapping[str, Any]) -> None:
    """Append an arbitrary event payload into the trace JSONL file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, default=_default_serializer)
        handle.write("\n")


__all__ = ["write_trace_jsonl", "append_trace_event"]
