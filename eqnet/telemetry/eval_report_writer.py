from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class EvalReportWriterConfig:
    telemetry_dir: Path


class EvalReportWriter:
    """Append-only JSONL writer for shadow/canary evaluation reports."""

    REQUIRED_KEYS = (
        "schema_version",
        "eval_report_id",
        "timestamp_ms",
        "proposal_id",
        "method",
        "verdict",
        "metrics_before",
        "metrics_after",
        "source_week",
    )

    VALID_VERDICTS = {"PASS", "FAIL", "INCONCLUSIVE"}

    def __init__(self, cfg: EvalReportWriterConfig) -> None:
        self._cfg = cfg

    def append(
        self,
        *,
        timestamp_ms: int,
        proposal_id: str,
        method: str,
        verdict: str,
        metrics_before: Mapping[str, Any],
        metrics_after: Mapping[str, Any],
        source_week: str,
        eval_report_id: str | None = None,
        delta: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        norm_verdict = str(verdict).strip().upper()
        if norm_verdict not in self.VALID_VERDICTS:
            raise ValueError(f"invalid verdict: {verdict}")
        pid = str(proposal_id).strip()
        if not pid:
            raise ValueError("proposal_id is required")

        record: dict[str, Any] = {
            "schema_version": "eval_report.v0",
            "eval_report_id": eval_report_id or str(uuid.uuid4()),
            "timestamp_ms": int(timestamp_ms),
            "proposal_id": pid,
            "method": str(method),
            "verdict": norm_verdict,
            "metrics_before": dict(metrics_before),
            "metrics_after": dict(metrics_after),
            "source_week": str(source_week),
        }
        if delta is not None:
            record["delta"] = dict(delta)
        if extra:
            for key, value in extra.items():
                if value is not None:
                    record[str(key)] = value

        for key in self.REQUIRED_KEYS:
            if key not in record:
                raise ValueError(f"missing required key: {key}")

        day = datetime.fromtimestamp(int(timestamp_ms) / 1000, tz=timezone.utc).strftime("%Y%m%d")
        out_path = self._cfg.telemetry_dir / f"eval_reports-{day}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return out_path

