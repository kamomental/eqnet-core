from __future__ import annotations

import json
from pathlib import Path

from eqnet.telemetry.sync_downshift_writer import SyncDownshiftWriter, SyncDownshiftWriterConfig


def test_sync_downshift_writer_emits_required_keys(tmp_path: Path) -> None:
    writer = SyncDownshiftWriter(SyncDownshiftWriterConfig(telemetry_dir=tmp_path))
    out = writer.append(
        timestamp_ms=1735689600000,
        reason_codes=["DOWNSHIFT_HARM_STREAK"],
        cooldown_until_ts_ms=1735689900000,
        actions=["STOP_SYNC_PROPOSALS"],
        policy_meta={"policy_version": "realtime_downshift_policy_v0"},
        source_week="2025-W01",
    )
    payload = json.loads(out.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert payload["schema_version"] == "sync_downshift_applied.v0"
    assert payload["cooldown_until_ts_ms"] == 1735689900000
