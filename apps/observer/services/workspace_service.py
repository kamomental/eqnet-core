from __future__ import annotations

from typing import Any

from ..stores.workspace_store import WorkspaceRecord, WorkspaceStore


class WorkspaceService:
    def __init__(self, store: WorkspaceStore) -> None:
        self.store = store

    def page(
        self,
        day: str,
        filename: str,
        *,
        offset: int,
        limit: int,
        turn_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        rows, next_offset = self.store.read_page(
            day,
            filename,
            offset=offset,
            limit=limit,
            turn_id=turn_id,
        )
        data = []
        for record in rows:
            payload = dict(record.payload)
            payload.setdefault("file", record.file)
            payload.setdefault("line_no", record.line_no)
            data.append(payload)
        return data, next_offset


__all__ = ["WorkspaceService"]
