from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SyncDownshiftWriterConfig:
    telemetry_dir: Path


class SyncDownshiftWriter:
    REQUIRED_KEYS = (
        "schema_version",
        "event_id",
        "timestamp_ms",
        "reason_codes",
        "cooldown_until_ts_ms",
        "actions",
        "policy_meta",
        "source_week",
    )

    def __init__(self, cfg: SyncDownshiftWriterConfig) -> None:
        self._cfg = cfg

    def append(
        self,
        *,
        timestamp_ms: int,
        reason_codes: list[str],
        cooldown_until_ts_ms: int,
        actions: list[str],
        policy_meta: Mapping[str, Any],
        source_week: str,
        event_id: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        record: dict[str, Any] = {
            "schema_version": "sync_downshift_applied.v0",
            "event_id": event_id or str(uuid.uuid4()),
            "timestamp_ms": int(timestamp_ms),
            "reason_codes": [str(x) for x in reason_codes],
            "cooldown_until_ts_ms": int(cooldown_until_ts_ms),
            "actions": [str(x) for x in actions],
            "policy_meta": dict(policy_meta),
            "source_week": str(source_week),
        }
        if extra:
            for key, value in extra.items():
                if value is not None:
                    record[str(key)] = value
        for key in self.REQUIRED_KEYS:
            if key not in record:
                raise ValueError(f"missing required key: {key}")
        day = datetime.fromtimestamp(int(timestamp_ms) / 1000, tz=timezone.utc).strftime("%Y%m%d")
        out_path = self._cfg.telemetry_dir / f"sync_downshifts-{day}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return out_path
