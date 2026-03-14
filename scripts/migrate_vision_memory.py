#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional


def _safe_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    text = str(value).strip()
    return text or None


def _entry_id(payload: Dict[str, Any], fallback_index: int) -> str:
    ts_value = payload.get("timestamp")
    try:
        ts_ms = int(float(ts_value) * 1000)
    except Exception:
        ts_ms = fallback_index
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    response_id = _safe_text(meta.get("response_id"))
    turn_id = int(payload.get("turn_id") or 0)
    if response_id:
        return f"vision-{ts_ms}-{response_id[-12:]}"
    return f"vision-{ts_ms}-{turn_id}"


def _normalize_row(row: Dict[str, Any], fallback_index: int) -> Dict[str, Any]:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    timestamp = row.get("timestamp")
    try:
        timestamp = float(timestamp)
    except Exception:
        timestamp = float(fallback_index)
    normalized = {
        "id": _safe_text(row.get("id")) or _entry_id(row, fallback_index),
        "schema": _safe_text(row.get("schema")) or "observed_vision/v1",
        "kind": _safe_text(row.get("kind")) or "observed",
        "modality": _safe_text(row.get("modality")) or "vision",
        "turn_id": int(row.get("turn_id") or 0),
        "session_id": row.get("session_id"),
        "timestamp": timestamp,
        "type": _safe_text(row.get("type")) or "vision_summary",
        "text": _safe_text(row.get("text")) or "",
        "summary": _safe_text(row.get("summary")) or _safe_text(row.get("text")) or "",
        "user_text": _safe_text(row.get("user_text")),
        "image_path": _safe_text(row.get("image_path")),
        "talk_mode": _safe_text(row.get("talk_mode")) or "talk",
        "response_route": _safe_text(row.get("response_route")) or "habit",
        "meta": {
            "model": _safe_text(meta.get("model")),
            "backend": _safe_text(meta.get("backend")),
            "response_id": _safe_text(meta.get("response_id")),
        },
    }
    normalized["id"] = _entry_id(normalized, fallback_index)
    return normalized


def migrate_file(path: Path, *, backup_suffix: str = ".bak") -> Dict[str, int]:
    if not path.exists():
        return {"rows": 0, "migrated": 0}

    backup_path = path.with_name(path.name + backup_suffix)
    shutil.copy2(path, backup_path)

    rows = []
    migrated = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            normalized = _normalize_row(payload, index)
            if normalized != payload:
                migrated += 1
            rows.append(normalized)

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"rows": len(rows), "migrated": migrated}


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize vision memory JSONL rows to observed_vision/v1.")
    parser.add_argument(
        "--path",
        default="logs/vision_memory.jsonl",
        help="Vision memory JSONL path.",
    )
    args = parser.parse_args()
    result = migrate_file(Path(args.path))
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
