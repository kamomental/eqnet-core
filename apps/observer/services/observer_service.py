from __future__ import annotations

from pathlib import Path
from typing import Any

from ..stores.audit_store import AuditStore
from ..stores.trace_store import TraceStore
from .redact import redact_trace_payload


class ObserverService:
    """Facade that exposes JSON-ready payloads for APIs and HTML views."""

    def __init__(self, audit_store: AuditStore, trace_store: TraceStore) -> None:
        self.audit_store = audit_store
        self.trace_store = trace_store

    # API -----------------------------------------------------------------
    def list_audits(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in self.audit_store.list_items():
            out.append(
                {
                    "date": item.date,
                    "health": {
                        "status": item.health_status,
                        "reasons": item.health_reasons,
                    },
                    "summary": {
                        "fatal_failures": item.fatal_failures,
                        "warn_fail_rate": item.warn_fail_rate,
                        "boundary": {
                            "max_length": item.boundary_max_length,
                        },
                        "top_reasons": item.top_reasons,
                    },
                }
            )
        return out

    def latest_audit_date(self) -> str | None:
        items = self.audit_store.list_items()
        return items[0].date if items else None

    def audit_file_path(self, date: str) -> Path:
        return self.audit_store.get_path(date)

    def read_audit(self, date: str) -> dict[str, Any]:
        path = self.audit_store.get_path(date)
        return self.audit_store.read_json(path)

    def list_traces(self, day: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in self.trace_store.list_files(day):
            meta = self.trace_store.peek_meta(day, path.name)
            out.append(
                {
                    "file": meta.file,
                    "source_loop": meta.source_loop,
                    "pid": meta.pid,
                    "size_bytes": meta.size,
                    "turn_range": {
                        "min": meta.turn_min,
                        "max": meta.turn_max,
                    },
                }
            )
        return out

    def read_trace_page(
        self,
        day: str,
        filename: str,
        *,
        offset: int,
        limit: int,
        turn_id: str | None = None,
        turn_id_contains: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        raw_rows, next_offset = self.trace_store.read_page(
            day,
            filename,
            offset=offset,
            limit=limit,
            turn_id=turn_id,
            turn_id_contains=turn_id_contains,
        )
        rows: list[dict[str, Any]] = []
        for line_no, payload in raw_rows:
            rows.append(
                {
                    "line_no": line_no,
                    "timestamp_ms": payload.get("timestamp_ms"),
                    "turn_id": payload.get("turn_id"),
                    "redacted": redact_trace_payload(payload),
                }
            )
        return rows, next_offset

    # Overlay --------------------------------------------------------------
    def overlay_model_for_date(self, date: str) -> dict[str, Any]:
        payload = self.read_audit(date)
        health = payload.get("health") or {}
        stats = payload.get("stats") or {}
        reasons = list(health.get("reasons") or [])[:3]
        return {
            "date": date,
            "status": health.get("status", "UNKNOWN"),
            "reasons": reasons,
            "fatal_failures": stats.get("fatal_failures", 0),
            "warn_fail_rate": stats.get("warn_fail_rate"),
        }

    def overlay_model_latest(self) -> dict[str, Any] | None:
        latest = self.latest_audit_date()
        if not latest:
            return None
        return self.overlay_model_for_date(latest)


__all__ = ["ObserverService"]
