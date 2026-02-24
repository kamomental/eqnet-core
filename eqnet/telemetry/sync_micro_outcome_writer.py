from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SyncMicroOutcomeWriterConfig:
    telemetry_dir: Path


class SyncMicroOutcomeWriter:
    REQUIRED_KEYS = (
        "schema_version",
        "outcome_id",
        "timestamp_ms",
        "proposal_id",
        "execution_id",
        "window_sec",
        "baseline_r",
        "observed_r",
        "delta_r",
        "result",
        "reason_codes",
        "evaluated_at_eval_ts_ms",
        "source_week",
    )
    VALID_RESULTS = {"HELPED", "NO_EFFECT", "HARMED", "UNKNOWN"}

    def __init__(self, cfg: SyncMicroOutcomeWriterConfig) -> None:
        self._cfg = cfg

    def append(
        self,
        *,
        timestamp_ms: int,
        proposal_id: str,
        execution_id: str,
        window_sec: int,
        baseline_r: float | None,
        observed_r: float | None,
        delta_r: float | None,
        result: str,
        reason_codes: list[str],
        evaluated_at_eval_ts_ms: int,
        source_week: str,
        outcome_id: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        norm_result = str(result).strip().upper()
        if norm_result not in self.VALID_RESULTS:
            raise ValueError(f"invalid result: {result}")
        if int(window_sec) <= 0:
            raise ValueError("window_sec must be > 0")
        record: dict[str, Any] = {
            "schema_version": "sync_micro_outcome.v0",
            "outcome_id": outcome_id or str(uuid.uuid4()),
            "timestamp_ms": int(timestamp_ms),
            "proposal_id": str(proposal_id),
            "execution_id": str(execution_id),
            "window_sec": int(window_sec),
            "baseline_r": float(baseline_r) if isinstance(baseline_r, (int, float)) else None,
            "observed_r": float(observed_r) if isinstance(observed_r, (int, float)) else None,
            "delta_r": float(delta_r) if isinstance(delta_r, (int, float)) else None,
            "result": norm_result,
            "reason_codes": [str(x) for x in reason_codes],
            "evaluated_at_eval_ts_ms": int(evaluated_at_eval_ts_ms),
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
        out_path = self._cfg.telemetry_dir / f"sync_micro_outcomes-{day}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return out_path
