"""Utility helpers for working with JSON Lines (jsonl) files."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Iterator, Sequence


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    """Append a JSON serialisable record to a jsonl file."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Yield dictionaries from a jsonl file if it exists."""
    if not os.path.exists(path):
        return iter(())
    def _iter() -> Iterator[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    return _iter()


def tail_jsonl(path: str, limit: int) -> Sequence[Dict[str, Any]]:
    """Return the last `limit` jsonl records (simple list)."""
    if not os.path.exists(path):
        return []
    buffer: list[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            buffer.append(obj)
            if len(buffer) > limit:
                buffer.pop(0)
    return buffer


__all__ = ["append_jsonl", "read_jsonl", "tail_jsonl"]
