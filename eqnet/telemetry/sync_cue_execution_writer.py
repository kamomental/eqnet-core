from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SyncCueExecutionWriterConfig:
    telemetry_dir: Path


class SyncCueExecutionWriter:
    """Append-only JSONL writer for approved sync cue executions."""

    REQUIRED_KEYS = (
        "schema_version",
        "execution_id",
        "timestamp_ms",
        "proposal_id",
        "cue_type",
        "ttl_sec",
        "status",
        "source_week",
    )

    VALID_STATUS = {"EXECUTED"}

    def __init__(self, cfg: SyncCueExecutionWriterConfig) -> None:
        self._cfg = cfg

    def append(
        self,
        *,
        timestamp_ms: int,
        proposal_id: str,
        cue_type: str,
        ttl_sec: int,
        source_week: str,
        status: str = "EXECUTED",
        execution_id: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        pid = str(proposal_id).strip()
        ctype = str(cue_type).strip()
        if not pid:
            raise ValueError("proposal_id is required")
        if not ctype:
            raise ValueError("cue_type is required")
        norm_status = str(status).strip().upper()
        if norm_status not in self.VALID_STATUS:
            raise ValueError(f"invalid status: {status}")
        ttl = int(ttl_sec)
        if ttl <= 0:
            raise ValueError("ttl_sec must be > 0")

        record: dict[str, Any] = {
            "schema_version": "sync_cue_execution.v0",
            "execution_id": execution_id or str(uuid.uuid4()),
            "timestamp_ms": int(timestamp_ms),
            "proposal_id": pid,
            "cue_type": ctype,
            "ttl_sec": ttl,
            "status": norm_status,
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
        out_path = self._cfg.telemetry_dir / f"sync_cue_executions-{day}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return out_path
