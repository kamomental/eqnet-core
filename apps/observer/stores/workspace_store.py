from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORKSPACE_EVENT = "workspace.snapshot_v1"


@dataclass(frozen=True)
class WorkspaceRecord:
    file: str
    line_no: int
    payload: dict[str, Any]

    @property
    def timestamp_ms(self) -> int | None:
        return self.payload.get("timestamp_ms")

    @property
    def turn_id(self) -> str | None:
        return self.payload.get("turn_id")

    @property
    def step(self) -> int | None:
        return self.payload.get("step")


class WorkspaceStore:
    def __init__(self, trace_root: Path) -> None:
        self.root = trace_root

    def _safe_day_dir(self, day: str) -> Path:
        day_dir = (self.root / day)
        if not day_dir.exists() or not day_dir.is_dir():
            raise FileNotFoundError(day_dir)
        return day_dir

    def list_files(self, day: str) -> list[Path]:
        day_dir = self._safe_day_dir(day)
        return sorted(day_dir.glob("*.jsonl"))

    def read_page(
        self,
        day: str,
        filename: str,
        *,
        offset: int,
        limit: int,
        turn_id: str | None = None,
    ) -> tuple[list[WorkspaceRecord], int]:
        day_dir = self._safe_day_dir(day)
        path = day_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)

        rows: list[WorkspaceRecord] = []
        next_offset = offset

        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if idx < offset:
                    continue
                if len(rows) >= limit:
                    break

                raw_line = line.strip()
                if not raw_line:
                    next_offset = idx + 1
                    continue
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError:
                    next_offset = idx + 1
                    continue

                if payload.get("event") != WORKSPACE_EVENT:
                    next_offset = idx + 1
                    continue

                if turn_id and payload.get("turn_id") != turn_id:
                    next_offset = idx + 1
                    continue

                record = WorkspaceRecord(file=filename, line_no=idx + 1, payload=payload)
                rows.append(record)
                next_offset = idx + 1

        return rows, next_offset


__all__ = ["WorkspaceStore", "WorkspaceRecord"]
