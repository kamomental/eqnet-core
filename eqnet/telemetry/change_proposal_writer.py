from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ChangeProposalWriterConfig:
    telemetry_dir: Path


class ChangeProposalWriter:
    """Append-only JSONL writer for change proposals."""

    REQUIRED_KEYS = (
        "schema_version",
        "proposal_id",
        "timestamp_ms",
        "trigger",
        "suggested_change",
        "expected_effect",
        "risk_level",
        "requires_gate",
        "source_week",
    )

    def __init__(self, cfg: ChangeProposalWriterConfig) -> None:
        self._cfg = cfg

    def append(
        self,
        *,
        timestamp_ms: int,
        trigger: Mapping[str, Any],
        suggested_change: Mapping[str, Any],
        expected_effect: Mapping[str, Any],
        risk_level: str,
        requires_gate: str,
        source_week: str,
        proposal_id: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        pid = proposal_id or str(uuid.uuid4())
        record: dict[str, Any] = {
            "schema_version": "change_proposal.v0",
            "proposal_id": pid,
            "timestamp_ms": int(timestamp_ms),
            "trigger": dict(trigger),
            "suggested_change": dict(suggested_change),
            "expected_effect": dict(expected_effect),
            "risk_level": str(risk_level),
            "requires_gate": str(requires_gate),
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
        out_path = self._cfg.telemetry_dir / f"change_proposals-{day}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return out_path

