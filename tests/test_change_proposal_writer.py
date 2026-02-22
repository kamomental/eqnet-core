from __future__ import annotations

import json
from pathlib import Path

from eqnet.telemetry.change_proposal_writer import (
    ChangeProposalWriter,
    ChangeProposalWriterConfig,
)


def test_change_proposal_writer_emits_required_keys(tmp_path: Path) -> None:
    writer = ChangeProposalWriter(ChangeProposalWriterConfig(telemetry_dir=tmp_path))
    out = writer.append(
        timestamp_ms=1735689600000,
        trigger={"kind": "mecpe_alert", "level": "WARN"},
        suggested_change={"action": "shadow_eval"},
        expected_effect={"primary": ["contract_errors_ratio_down"]},
        risk_level="LOW",
        requires_gate="shadow",
        source_week="2025-W01",
        proposal_id="00000000-0000-0000-0000-000000000000",
    )
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8").strip().splitlines()[-1])
    for key in [
        "schema_version",
        "proposal_id",
        "timestamp_ms",
        "trigger",
        "suggested_change",
        "expected_effect",
        "risk_level",
        "requires_gate",
        "source_week",
    ]:
        assert key in payload
    assert payload["schema_version"] == "change_proposal.v0"
    assert payload["proposal_id"] == "00000000-0000-0000-0000-000000000000"

