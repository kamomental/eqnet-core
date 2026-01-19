from __future__ import annotations

import json
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class Event:
    schema_version: str
    event_id: str
    event_type: str
    ts_unix_ms: int
    tz: str
    trace: Dict[str, Any]
    payload: Dict[str, Any]
    raw: Dict[str, Any]

    @staticmethod
    def from_json_line(line: str) -> "Event":
        obj = json.loads(line)
        t = obj.get("t") or {}
        trace = obj.get("trace") or {}
        payload = obj.get("payload") or {}
        return Event(
            schema_version=str(obj.get("schema_version", "")),
            event_id=str(obj.get("event_id", "")),
            event_type=str(obj.get("event_type", "")),
            ts_unix_ms=int(t.get("ts_unix_ms", 0)),
            tz=str(t.get("tz", "")),
            trace=trace,
            payload=payload,
            raw=obj,
        )


@dataclass(frozen=True)
class DerivedMetricsPayload:
    window_ms: int
    end_ts_unix_ms: int
    metrics: Dict[str, Optional[float]]
    sources: Dict[str, Any]


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_json_dumps(obj: Any) -> bytes:
    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def compute_payload_hash(event_obj: Dict[str, Any]) -> str:
    return sha256_hex(stable_json_dumps(event_obj))


def now_iso_utc_from_unix_ms(ts_unix_ms: int) -> str:
    dt = datetime.fromtimestamp(ts_unix_ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%SZ")


def normalize_hash_list(hashes: Iterable[str]) -> List[str]:
    return sorted([h for h in hashes if isinstance(h, str)])


def short_hash_from_event_hashes(event_hashes: List[str], length: int = 12) -> str:
    norm = normalize_hash_list(event_hashes)
    h = sha256_hex(stable_json_dumps(norm))
    return h[:length]


def make_dm_event_id(end_ts_unix_ms: int, window_label: str, event_hashes: List[str]) -> str:
    iso = now_iso_utc_from_unix_ms(end_ts_unix_ms)
    sh = short_hash_from_event_hashes(event_hashes)
    return f"dm_{iso}_{window_label}_{sh}"


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def read_last_payload_hash(jsonl_path: str, max_bytes: int = 65536) -> str:
    if not os.path.exists(jsonl_path):
        return ""
    size = os.path.getsize(jsonl_path)
    if size <= 0:
        return ""
    with open(jsonl_path, "rb") as f:
        seek_pos = max(0, size - max_bytes)
        f.seek(seek_pos)
        data = f.read()
    lines = data.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if not line:
            continue
        try:
            obj = json.loads(line.decode("utf-8"))
            integrity = (obj.get("trace") or {}).get("integrity") or {}
            ph = integrity.get("payload_hash")
            return ph if isinstance(ph, str) else ""
        except Exception:
            continue
    return ""


def reason_code_hash(code: str) -> str:
    return sha256_hex(code.encode("utf-8"))[:16]
