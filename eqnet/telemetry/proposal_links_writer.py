from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Any


@dataclass(frozen=True)
class ProposalLinkWriterConfig:
    telemetry_dir: Path


class ProposalLinkWriter:
    """Append-only JSONL writer for proposal-to-artifact links."""

    REQUIRED_KEYS = (
        "schema_version",
        "link_id",
        "timestamp_ms",
        "proposal_id",
        "eval_report_id",
        "link_type",
        "source_week",
    )

    VALID_LINK_TYPES = {"shadow_eval", "canary_eval", "rollout_eval"}

    def __init__(self, cfg: ProposalLinkWriterConfig) -> None:
        self._cfg = cfg

    def append(
        self,
        *,
        timestamp_ms: int,
        proposal_id: str,
        eval_report_id: str,
        link_type: str,
        source_week: str,
        link_id: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        norm_link_type = str(link_type).strip()
        if norm_link_type not in self.VALID_LINK_TYPES:
            raise ValueError(f"invalid link_type: {link_type}")
        pid = str(proposal_id).strip()
        eid = str(eval_report_id).strip()
        if not pid:
            raise ValueError("proposal_id is required")
        if not eid:
            raise ValueError("eval_report_id is required")

        record: dict[str, Any] = {
            "schema_version": "proposal_link.v0",
            "link_id": link_id or str(uuid.uuid4()),
            "timestamp_ms": int(timestamp_ms),
            "proposal_id": pid,
            "eval_report_id": eid,
            "link_type": norm_link_type,
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
        out_path = self._cfg.telemetry_dir / f"proposal_links-{day}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return out_path

