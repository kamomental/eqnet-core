from __future__ import annotations

import json
from pathlib import Path

from apps.observer.stores.trace_store import TraceStore


def _write_trace(path: Path, rows: list[dict[str, object] | str]) -> None:
    serialised: list[str] = []
    for row in rows:
        if isinstance(row, str):
            serialised.append(row)
        else:
            serialised.append(json.dumps(row))
    path.write_text("\n".join(serialised), encoding="utf-8")


def test_trace_store_peek_meta(tmp_path: Path) -> None:
    root = tmp_path / "trace_v1"
    day = root / "2025-12-13"
    day.mkdir(parents=True)
    trace_path = day / "runtime-abc-1234.jsonl"
    _write_trace(
        trace_path,
        [
            {"timestamp_ms": 1, "turn_id": "1", "pid": 1234, "source_loop": "runtime"},
            {"timestamp_ms": 2, "turn_id": "250"},
            "{\"timestamp_ms\": 3",  # broken JSON line is ignored
        ],
    )

    store = TraceStore(root)
    meta = store.peek_meta("2025-12-13", trace_path.name)

    assert meta.file == trace_path.name
    assert meta.size == trace_path.stat().st_size
    assert meta.pid == 1234
    assert meta.turn_min == 1
    assert meta.turn_max == 250


def test_trace_store_paging_filters_and_invalid_lines(tmp_path: Path) -> None:
    root = tmp_path / "trace_v1"
    day = root / "2025-12-13"
    day.mkdir(parents=True)
    trace_path = day / "foo.jsonl"
    _write_trace(
        trace_path,
        [
            {"timestamp_ms": 1, "turn_id": "A"},
            {"timestamp_ms": 2, "turn_id": "B"},
            "{\"turn_id\": \"bad",
            {"timestamp_ms": 3, "turn_id": "AB"},
        ],
    )

    store = TraceStore(root)
    rows, next_offset = store.read_page("2025-12-13", "foo.jsonl", offset=0, limit=2)
    assert [line for line, _ in rows] == [1, 2]
    assert next_offset == 2

    rows2, next_offset2 = store.read_page("2025-12-13", "foo.jsonl", offset=next_offset, limit=10)
    assert [line for line, _ in rows2] == [4]
    assert next_offset2 == 4

    rows3, _ = store.read_page("2025-12-13", "foo.jsonl", offset=0, limit=10, turn_id="B")
    assert [payload["turn_id"] for _, payload in rows3] == ["B"]

    rows4, _ = store.read_page("2025-12-13", "foo.jsonl", offset=0, limit=10, turn_id_contains="A")
    assert [payload["turn_id"] for _, payload in rows4] == ["A", "AB"]
