from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditListItem:
    date: str
    health_status: str
    health_reasons: list[dict[str, Any]]
    fatal_failures: int
    warn_fail_rate: float | None
    boundary_max_length: int | None
    top_reasons: list[dict[str, Any]]
    memory_hint_category_topk: list[list[Any]]
    memory_hint_category_blocked_reason: dict[str, dict[str, Any]]


class AuditStore:
    """Filesystem-backed loader for nightly audit JSON blobs."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._cache: dict[Path, tuple[float, dict[str, Any]]] = {}

    def list_paths(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(self.root.glob("nightly_audit_*.json"))

    def get_path(self, date: str) -> Path:
        return self.root / f"nightly_audit_{date}.json"

    def read_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(path)
        mtime = path.stat().st_mtime
        cached = self._cache.get(path)
        if cached and cached[0] == mtime:
            return cached[1]
        data = json.loads(path.read_text(encoding="utf-8"))
        self._cache[path] = (mtime, data)
        return data

    def list_items(self) -> list[AuditListItem]:
        items: list[AuditListItem] = []
        for path in self.list_paths():
            payload = self.read_json(path)
            audit_payload = payload.get("nightly_audit") if isinstance(payload, dict) else None
            if not isinstance(audit_payload, dict):
                audit_payload = payload
            metadata = payload.get("metadata") or {}
            date = str(metadata.get("date") or path.stem.split("_")[-1])
            health = payload.get("health") or {}
            stats = payload.get("stats") or {}
            boundary = payload.get("boundary") or {}
            boundary_summary = boundary.get("summary") or {}
            reasons = list(health.get("reasons") or [])
            qualia_stats = audit_payload.get("qualia_gate_stats") or {}
            memory_hint = qualia_stats.get("memory_hint") if isinstance(qualia_stats, dict) else {}
            category_topk = list(memory_hint.get("memory_hint_category_topk") or [])
            category_blocked_reason = (
                memory_hint.get("memory_hint_category_blocked_reason") or {}
            )

            items.append(
                AuditListItem(
                    date=date,
                    health_status=str(health.get("status", "UNKNOWN")),
                    health_reasons=reasons[:3],
                    fatal_failures=int(stats.get("fatal_failures", 0) or 0),
                    warn_fail_rate=stats.get("warn_fail_rate"),
                    boundary_max_length=boundary_summary.get("max_length"),
                    top_reasons=reasons[:3],
                    memory_hint_category_topk=category_topk,
                    memory_hint_category_blocked_reason=category_blocked_reason,
                )
            )
        items.sort(key=lambda entry: entry.date, reverse=True)
        return items


__all__ = ["AuditListItem", "AuditStore"]
