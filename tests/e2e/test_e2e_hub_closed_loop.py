from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from eqnet.hub.api import EQNetConfig, EQNetHub
from tests.e2e.proof_utils import load_proof_scenario, prepare_event


from types import SimpleNamespace

def _as_moment_object(event: dict) -> object:
    payload = dict(event)
    timestamp_ms = payload.get("timestamp_ms", 0)
    payload["timestamp"] = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
    return SimpleNamespace(**payload)

def test_hub_closed_loop_generates_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = load_proof_scenario()
    config = EQNetConfig(
        telemetry_dir=tmp_path / "telemetry",
        reports_dir=tmp_path / "reports",
        state_dir=tmp_path / "state",
        trace_dir=tmp_path / "trace_v1",
        audit_dir=tmp_path / "audit",
    )
    hub = EQNetHub(config=config, embed_text_fn=lambda text: [])

    monkeypatch.setenv("EQNET_TRACE_V1", "1")
    monkeypatch.setenv("EQNET_TRACE_V1_DIR", str(config.trace_dir))

    for idx, raw in enumerate(scenario.events):
        event = prepare_event(raw, idx, scenario.scenario_id, scenario.date)
        moment = _as_moment_object(event)
        hub.log_moment(moment, event.get("user_text", "<redacted>"))

    hub.run_nightly(scenario.date)

    day = scenario.date.strftime("%Y-%m-%d")
    day_dir = config.trace_dir / day
    trace_files = list(day_dir.glob("hub-*.jsonl"))
    assert trace_files, "trace_v1 output should exist"

    audit_path = config.audit_dir / f"nightly_audit_{day}.json"
    assert audit_path.exists(), "audit artifact should be generated"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["schema_version"] == "nightly_audit_v1"
    health = audit.get("health") or {}
    assert health.get("status") in {"GREEN", "YELLOW", "RED"}
    assert isinstance(health.get("reasons"), list)
    evidence = audit.get("evidence") or {}
    assert "boundary_spans" in evidence
    assert "invariants" in evidence

