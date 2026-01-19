from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, TextIO

from common import compute_payload_hash, make_dm_event_id


@dataclass(frozen=True)
class EmitterConfig:
    schema_version: str
    tz: str
    session_id: str
    run_id: str


class Emitter:
    """Append derived_metrics events to a JSONL file."""

    def __init__(self, cfg: EmitterConfig) -> None:
        self._cfg = cfg
        self._prev_hash: str = ""

    def set_prev_hash(self, prev_hash: str) -> None:
        self._prev_hash = prev_hash or ""

    def emit_derived_metrics(
        self,
        fp_out: TextIO,
        *,
        window_label: str,
        payload: Dict[str, Any],
        end_ts_unix_ms: int,
    ) -> str:
        sources = payload.get("sources") or {}
        event_hashes = sources.get("event_hashes") or []
        dm_event_id = make_dm_event_id(
            end_ts_unix_ms=end_ts_unix_ms,
            window_label=window_label,
            event_hashes=event_hashes,
        )

        event_obj: Dict[str, Any] = {
            "schema_version": self._cfg.schema_version,
            "event_id": dm_event_id,
            "event_type": "derived_metrics",
            "t": {"ts_unix_ms": end_ts_unix_ms, "tz": self._cfg.tz},
            "trace": {
                "session_id": self._cfg.session_id,
                "run_id": self._cfg.run_id,
                "actor": "system",
                "integrity": {"payload_hash": "", "prev_hash": self._prev_hash},
            },
            "payload": payload,
        }

        payload_hash = compute_payload_hash(event_obj)
        event_obj["trace"]["integrity"]["payload_hash"] = payload_hash

        fp_out.write(json.dumps(event_obj, ensure_ascii=False, separators=(",", ":")) + "\n")
        fp_out.flush()

        self._prev_hash = payload_hash
        return payload_hash


@dataclass(frozen=True)
class AuditEmitterConfig:
    schema_version: str
    tz: str
    session_id: str
    run_id: str


class AuditEmitter:
    """Append control_audit events to a JSONL file."""

    def __init__(self, cfg: AuditEmitterConfig) -> None:
        self._cfg = cfg
        self._prev_hash: str = ""

    def set_prev_hash(self, prev_hash: str) -> None:
        self._prev_hash = prev_hash or ""

    def emit_control_audit(
        self,
        fp_out: TextIO,
        *,
        ts_unix_ms: int,
        action: str,
        scope: Dict[str, Any],
        reason_hash: str,
    ) -> str:
        event_obj: Dict[str, Any] = {
            "schema_version": self._cfg.schema_version,
            "event_id": f"ca_{ts_unix_ms}_{action}",
            "event_type": "control_audit",
            "t": {"ts_unix_ms": ts_unix_ms, "tz": self._cfg.tz},
            "trace": {
                "session_id": self._cfg.session_id,
                "run_id": self._cfg.run_id,
                "actor": "system",
                "integrity": {"payload_hash": "", "prev_hash": self._prev_hash},
            },
            "payload": {
                "kind": "control_audit",
                "action": action,
                "scope": scope,
                "reason_hash": reason_hash,
            },
        }

        payload_hash = compute_payload_hash(event_obj)
        event_obj["trace"]["integrity"]["payload_hash"] = payload_hash

        fp_out.write(json.dumps(event_obj, ensure_ascii=False, separators=(",", ":")) + "\n")
        fp_out.flush()

        self._prev_hash = payload_hash
        return payload_hash
