from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def resolve_trace_files(trace_path: Path) -> List[Path]:
    if trace_path.is_file():
        return [trace_path]
    if not trace_path.exists():
        return []
    files = sorted(p for p in trace_path.rglob("*.jsonl") if p.is_file())
    return files


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows

