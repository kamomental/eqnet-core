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
