from __future__ import annotations

import json
from pathlib import Path

import pytest

from eqnet.telemetry.sync_micro_outcome_writer import (
    SyncMicroOutcomeWriter,
    SyncMicroOutcomeWriterConfig,
)


def test_sync_micro_outcome_writer_emits_required_keys(tmp_path: Path) -> None:
    writer = SyncMicroOutcomeWriter(SyncMicroOutcomeWriterConfig(telemetry_dir=tmp_path))
    out = writer.append(
        timestamp_ms=1735689600000,
        proposal_id="sync-1",
        execution_id="exec-1",
        window_sec=60,
        baseline_r=0.3,
        observed_r=0.4,
        delta_r=0.1,
        result="HELPED",
        reason_codes=["SYNC_HELPED_R_UP"],
        evaluated_at_eval_ts_ms=1735689660000,
        source_week="2025-W01",
    )
    payload = json.loads(out.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert payload["schema_version"] == "sync_micro_outcome.v0"
    assert payload["execution_id"] == "exec-1"
    assert payload["result"] == "HELPED"


def test_sync_micro_outcome_writer_rejects_invalid_result(tmp_path: Path) -> None:
    writer = SyncMicroOutcomeWriter(SyncMicroOutcomeWriterConfig(telemetry_dir=tmp_path))
    with pytest.raises(ValueError):
        writer.append(
            timestamp_ms=1735689600000,
            proposal_id="sync-1",
            execution_id="exec-1",
            window_sec=60,
            baseline_r=0.3,
            observed_r=0.4,
            delta_r=0.1,
            result="BAD",
            reason_codes=[],
            evaluated_at_eval_ts_ms=1735689660000,
            source_week="2025-W01",
        )
