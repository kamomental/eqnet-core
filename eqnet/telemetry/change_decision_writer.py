from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ChangeDecisionWriterConfig:
    telemetry_dir: Path


class ChangeDecisionWriter:
    """Append-only JSONL writer for change decisions."""

    REQUIRED_KEYS = (
        "schema_version",
        "decision_id",
        "timestamp_ms",
        "proposal_id",
        "decision",
        "actor",
        "reason",
        "source_week",
    )

    VALID_DECISIONS = {
        "REJECT",
        "ACCEPT_SHADOW",
        "ACCEPT_CANARY",
        "ACCEPT_ROLLOUT",
        "ROLLBACK",
        "LINK_EVAL_REPORT",
    }

    VALID_ACTORS = {"auto", "human"}

    def __init__(self, cfg: ChangeDecisionWriterConfig) -> None:
        self._cfg = cfg

    def append(
        self,
        *,
        timestamp_ms: int,
        proposal_id: str,
        decision: str,
        actor: str,
        reason: str,
        source_week: str,
        decision_id: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        norm_decision = str(decision).strip().upper()
        norm_actor = str(actor).strip().lower()
        if norm_decision not in self.VALID_DECISIONS:
            raise ValueError(f"invalid decision: {decision}")
        if norm_actor not in self.VALID_ACTORS:
            raise ValueError(f"invalid actor: {actor}")
        pid = str(proposal_id).strip()
        if not pid:
            raise ValueError("proposal_id is required")

        record: dict[str, Any] = {
            "schema_version": "change_decision.v0",
            "decision_id": decision_id or str(uuid.uuid4()),
            "timestamp_ms": int(timestamp_ms),
            "proposal_id": pid,
            "decision": norm_decision,
            "actor": norm_actor,
            "reason": str(reason),
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
        out_path = self._cfg.telemetry_dir / f"change_decisions-{day}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return out_path
