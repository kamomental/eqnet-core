from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from eqnet.telemetry.change_decision_writer import (
    ChangeDecisionWriter,
    ChangeDecisionWriterConfig,
)


def test_change_decision_writer_emits_required_keys(tmp_path: Path) -> None:
    writer = ChangeDecisionWriter(ChangeDecisionWriterConfig(telemetry_dir=tmp_path))
    out = writer.append(
        timestamp_ms=1735689600000,
        proposal_id="00000000-0000-0000-0000-000000000001",
        decision="ACCEPT_SHADOW",
        actor="human",
        reason="manual review approved for shadow",
        source_week="2025-W01",
        decision_id="00000000-0000-0000-0000-000000000002",
    )
    payload = json.loads(out.read_text(encoding="utf-8").strip().splitlines()[-1])
    for key in [
        "schema_version",
        "decision_id",
        "timestamp_ms",
        "proposal_id",
        "decision",
        "actor",
        "reason",
        "source_week",
    ]:
        assert key in payload
    assert payload["schema_version"] == "change_decision.v0"
    assert payload["decision"] == "ACCEPT_SHADOW"
    assert payload["actor"] == "human"


def test_change_decision_writer_rejects_invalid_decision(tmp_path: Path) -> None:
    writer = ChangeDecisionWriter(ChangeDecisionWriterConfig(telemetry_dir=tmp_path))
    with pytest.raises(ValueError):
        writer.append(
            timestamp_ms=1735689600000,
            proposal_id="p-1",
            decision="INVALID",
            actor="human",
            reason="bad",
            source_week="2025-W01",
        )


def test_record_change_decision_cli_writes_jsonl(tmp_path: Path) -> None:
    telemetry_dir = tmp_path / "telemetry"
    cmd = [
        sys.executable,
        "scripts/record_change_decision.py",
        "--telemetry_dir",
        str(telemetry_dir),
        "--proposal_id",
        "00000000-0000-0000-0000-000000000003",
        "--decision",
        "REJECT",
        "--reason",
        "manual rejection",
        "--actor",
        "human",
        "--timestamp_ms",
        "1735689600000",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    files = sorted(telemetry_dir.glob("change_decisions-*.jsonl"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8").strip().splitlines()[-1])
    assert payload["proposal_id"] == "00000000-0000-0000-0000-000000000003"
    assert payload["decision"] == "REJECT"

