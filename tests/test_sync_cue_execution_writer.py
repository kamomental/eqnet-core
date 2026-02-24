from __future__ import annotations

import json
from pathlib import Path

import pytest

from eqnet.telemetry.sync_cue_execution_writer import (
    SyncCueExecutionWriter,
    SyncCueExecutionWriterConfig,
)


def test_sync_cue_execution_writer_emits_required_keys(tmp_path: Path) -> None:
    writer = SyncCueExecutionWriter(SyncCueExecutionWriterConfig(telemetry_dir=tmp_path))
    out = writer.append(
        timestamp_ms=1735689600000,
        proposal_id="sync-1",
        cue_type="CUE_BREATH_INHALE_ON_BEAT",
        ttl_sec=10,
        source_week="2025-W01",
        execution_id="00000000-0000-0000-0000-000000000001",
    )
    payload = json.loads(out.read_text(encoding="utf-8").strip().splitlines()[-1])
    for key in [
        "schema_version",
        "execution_id",
        "timestamp_ms",
        "proposal_id",
        "cue_type",
        "ttl_sec",
        "status",
        "source_week",
    ]:
        assert key in payload
    assert payload["schema_version"] == "sync_cue_execution.v0"
    assert payload["status"] == "EXECUTED"


def test_sync_cue_execution_writer_rejects_invalid_ttl(tmp_path: Path) -> None:
    writer = SyncCueExecutionWriter(SyncCueExecutionWriterConfig(telemetry_dir=tmp_path))
    with pytest.raises(ValueError):
        writer.append(
            timestamp_ms=1735689600000,
            proposal_id="sync-1",
            cue_type="CUE_BREATH_INHALE_ON_BEAT",
            ttl_sec=0,
            source_week="2025-W01",
        )
