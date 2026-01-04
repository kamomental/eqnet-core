from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from eqnet.hub.api import EQNetConfig, EQNetHub


def test_run_nightly_audit_writes_file(tmp_path):
    trace_root = tmp_path / "trace_v1"
    day = "2025-12-14"
    day_dir = trace_root / day
    day_dir.mkdir(parents=True, exist_ok=True)
    sample = {
        "timestamp_ms": 1,
        "turn_id": "turn-1",
        "scenario_id": "audit-demo",
        "source_loop": "hub",
        "boundary": {"score": 0.7},
        "prospection": {"accepted": True},
        "policy": {"throttles": {}},
        "invariants": {"TRACE_001": True},
        "ru_v0": {
            "gate_action": "EXECUTE",
            "policy_version": "ru-v0.1",
            "missing_required_fields": [],
        },
    }
    (day_dir / "hub-1.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")

    config = EQNetConfig(
        telemetry_dir=tmp_path / "telemetry",
        reports_dir=tmp_path / "reports",
        state_dir=tmp_path / "state",
        trace_dir=trace_root,
        audit_dir=tmp_path / "audit",
    )
    hub = EQNetHub(config=config, embed_text_fn=lambda text: [])

    hub._run_nightly_audit(date(2025, 12, 14))
    audit_path = tmp_path / "audit" / f"nightly_audit_{day}.json"
    assert audit_path.exists()
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert "ru_v0_summary" in payload
    ru = payload["ru_v0_summary"]
    assert isinstance(ru, dict)
    assert "gate_action_counts" in ru
    assert isinstance(ru["gate_action_counts"], dict)
    assert "policy_version_counts" in ru
    assert isinstance(ru["policy_version_counts"], dict)
    assert "ru-v0.1" in ru["policy_version_counts"]
    assert "missing_required_fields_events" in ru
    assert isinstance(ru["missing_required_fields_events"], int)
    assert "ru_v0_events" in ru
    assert isinstance(ru["ru_v0_events"], int)
