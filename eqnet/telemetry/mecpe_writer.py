from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _coerce_sha256(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        return hashlib.sha256(bytes(value)).hexdigest()
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    if len(lowered) == 64 and all(c in "0123456789abcdef" for c in lowered):
        return lowered
    return _sha256_hex(text)


@dataclass(frozen=True)
class MecpeWriterConfig:
    telemetry_dir: Path


class MecpeWriter:
    """Append-only JSONL writer for MECPE telemetry records."""

    def __init__(self, cfg: MecpeWriterConfig) -> None:
        self._cfg = cfg

    def append_turn(
        self,
        *,
        timestamp_ms: int,
        turn_id: str,
        prompt_hash: str,
        model_version: str,
        text_hash: str,
        audio_sha256: str,
        video_sha256: str,
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        record: dict[str, Any] = {
            "schema_version": "mecpe_record.v0",
            "timestamp_ms": int(timestamp_ms),
            "turn_id": str(turn_id),
            "prompt_hash": str(prompt_hash),
            "model": {"version": str(model_version)},
            "text_hash": str(text_hash),
            "audio_sha256": str(audio_sha256),
            "video_sha256": str(video_sha256),
        }
        if extra:
            for key, value in extra.items():
                if value is not None:
                    record[str(key)] = value

        day = datetime.fromtimestamp(int(timestamp_ms) / 1000, tz=timezone.utc).strftime("%Y%m%d")
        out_path = self._cfg.telemetry_dir / f"mecpe-{day}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return out_path


def _mapping_value(raw_event: Any, *keys: str) -> Any:
    if isinstance(raw_event, Mapping):
        for key in keys:
            if key in raw_event:
                return raw_event.get(key)
        return None
    for key in keys:
        if hasattr(raw_event, key):
            return getattr(raw_event, key)
    return None


def build_minimal_mecpe_payload(
    *,
    timestamp_ms: int,
    turn_id: str,
    raw_text: str,
    raw_event: Any,
    text_hash_override: str | None = None,
    prompt_seed: str = "mecpe:dummy:v0",
    model_version: str = "mecpe-dummy-v0",
) -> dict[str, Any]:
    text_hash = _coerce_sha256(text_hash_override) or _sha256_hex(raw_text or "")
    prompt_hash = _sha256_hex(prompt_seed)
    audio_sha256 = _coerce_sha256(_mapping_value(raw_event, "audio_sha256", "audio_hash"))
    video_sha256 = _coerce_sha256(_mapping_value(raw_event, "video_sha256", "video_hash"))
    return {
        "timestamp_ms": int(timestamp_ms),
        "turn_id": str(turn_id),
        "prompt_hash": prompt_hash,
        "model_version": str(model_version),
        "text_hash": text_hash,
        "audio_sha256": audio_sha256,
        "video_sha256": video_sha256,
    }
