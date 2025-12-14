from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceMeta:
    file: str
    source_loop: str | None
    pid: int | None
    size: int
    turn_min: int | None
    turn_max: int | None


class TraceStore:
    """Filesystem helper that lists trace files and streams JSONL pages."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._meta_cache: dict[Path, tuple[float, TraceMeta]] = {}

    # -- path helpers -----------------------------------------------------
    def _safe_day_dir(self, day: str) -> Path:
        day_path = (self.root / day).resolve()
        root = self.root.resolve()
        if root not in day_path.parents and day_path != root:
            raise FileNotFoundError("invalid trace day")
        return day_path

    def _safe_file_path(self, day: str, filename: str) -> Path:
        if Path(filename).name != filename:
            raise FileNotFoundError("invalid trace filename")
        day_dir = self._safe_day_dir(day)
        file_path = (day_dir / filename).resolve()
        if day_dir not in file_path.parents:
            raise FileNotFoundError("invalid trace file path")
        return file_path

    # -- listing ----------------------------------------------------------
    def list_files(self, day: str) -> list[Path]:
        day_dir = self._safe_day_dir(day)
        if not day_dir.exists():
            return []
        return sorted(day_dir.glob("*.jsonl"))

    # -- metadata ---------------------------------------------------------
    def _read_first_jsonl(self, path: Path) -> dict[str, Any] | None:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                data = line.strip()
                if not data:
                    continue
                try:
                    return json.loads(data)
                except json.JSONDecodeError:
                    continue
        return None

    def _read_last_jsonl(self, path: Path, chunk_size: int = 64 * 1024) -> dict[str, Any] | None:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            file_end = handle.tell()
            if file_end == 0:
                return None

            pos = file_end
            leftover = b""
            while pos > 0:
                read_len = min(chunk_size, pos)
                pos -= read_len
                handle.seek(pos)
                data = handle.read(read_len)
                leftover = data + leftover
                rows = leftover.splitlines()
                for raw in reversed(rows):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        return json.loads(raw.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
            return None

    def _coerce_turn_int(self, payload: dict[str, Any] | None) -> int | None:
        if not payload:
            return None
        keys = ("turn", "turn_index", "turn_no", "turn_number")
        for key in keys:
            value = payload.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
        value = payload.get("turn_id")
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return None

    def peek_meta(self, day: str, filename: str) -> TraceMeta:
        path = self._safe_file_path(day, filename)
        mtime = path.stat().st_mtime
        cached = self._meta_cache.get(path)
        if cached and cached[0] == mtime:
            return cached[1]

        first = self._read_first_jsonl(path)
        last = self._read_last_jsonl(path)
        size = path.stat().st_size

        source_loop: str | None = None
        pid: int | None = None
        if isinstance(first, dict):
            source_loop = (first.get("source_loop") or first.get("loop") or first.get("loop_name"))
            pid_value = first.get("pid")
            if isinstance(pid_value, int):
                pid = pid_value
            elif isinstance(pid_value, str) and pid_value.isdigit():
                pid = int(pid_value)

        if (source_loop is None or pid is None) and "-" in filename:
            stem = filename[:-6] if filename.endswith(".jsonl") else filename
            parts = stem.split("-")
            if source_loop is None and parts:
                source_loop = parts[0]
            if pid is None:
                for part in reversed(parts):
                    if part.isdigit():
                        pid = int(part)
                        break

        meta = TraceMeta(
            file=filename,
            source_loop=source_loop,
            pid=pid,
            size=size,
            turn_min=self._coerce_turn_int(first),
            turn_max=self._coerce_turn_int(last),
        )
        self._meta_cache[path] = (mtime, meta)
        return meta

    # -- paging -----------------------------------------------------------
    def read_page(
        self,
        day: str,
        filename: str,
        *,
        offset: int,
        limit: int,
        turn_id: str | None = None,
        turn_id_contains: str | None = None,
    ) -> tuple[list[tuple[int, dict[str, Any]]], int]:
        path = self._safe_file_path(day, filename)
        rows: list[tuple[int, dict[str, Any]]] = []
        next_offset = offset

        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if idx < offset:
                    continue
                if len(rows) >= limit:
                    break
                data = line.strip()
                if not data:
                    next_offset = idx + 1
                    continue
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    next_offset = idx + 1
                    continue
                tid = payload.get("turn_id")
                if turn_id is not None and tid != turn_id:
                    next_offset = idx + 1
                    continue
                if turn_id_contains is not None and (tid is None or turn_id_contains not in str(tid)):
                    next_offset = idx + 1
                    continue
                rows.append((idx + 1, payload))
                next_offset = idx + 1

        return rows, next_offset


__all__ = ["TraceMeta", "TraceStore"]
