from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from eqnet.hub.api import EQNetConfig, EQNetHub
from eqnet.hub.idempotency import InMemoryIdempotencyStore


class _DelegateRuntime:
    def __init__(self) -> None:
        self.calls = []

    def log_moment(self, raw_event, raw_text, *, idempotency_key=None):  # noqa: ANN001, ARG002
        return None

    def run_nightly(self, date_obj=None, *, idempotency_key=None):  # noqa: ANN001
        self.calls.append(
            {
                "date": str(date_obj) if date_obj is not None else None,
                "idempotency_key": idempotency_key,
            }
        )

    def query_state(self, *, as_of=None):  # noqa: ANN001, ARG002
        return {}


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


def test_run_nightly_shadow_delegate_writes_trace_observation(tmp_path, monkeypatch):
    trace_root = tmp_path / "trace_v1"
    monkeypatch.setenv("EQNET_TRACE_V1", "1")
    monkeypatch.setenv("EQNET_TRACE_V1_DIR", str(trace_root))
    monkeypatch.setenv("HUB_DELEGATION_MODE", "shadow")

    delegate = _DelegateRuntime()
    config = EQNetConfig(
        telemetry_dir=tmp_path / "telemetry",
        reports_dir=tmp_path / "reports",
        state_dir=tmp_path / "state",
        trace_dir=trace_root,
        audit_dir=tmp_path / "audit",
    )
    hub = EQNetHub(config=config, embed_text_fn=lambda text: [], runtime_delegate=delegate)
    hub.run_nightly(date(2025, 12, 14))

    assert delegate.calls
    day_dir = trace_root / "2025-12-14"
    files = list(day_dir.glob("hub-*.jsonl"))
    assert files
    records = [
        json.loads(line)
        for line in files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    nightly_records = [
        r
        for r in records
        if r.get("policy", {}).get("observations", {}).get("hub", {}).get("operation") == "run_nightly"
    ]
    assert nightly_records
    obs = nightly_records[-1]["policy"]["observations"]["hub"]
    assert obs.get("delegation_mode") == "shadow"
    assert obs.get("delegate_status") == "ok"
    assert obs.get("idempotency_status") == "done"
    qobs = nightly_records[-1]["qualia"]["observations"]["hub"]
    assert "life_indicator_fingerprint" in qobs
    assert "policy_prior_fingerprint" in qobs
    assert "audit_fingerprint" in qobs


def test_run_nightly_idempotency_skip(tmp_path, monkeypatch):
    trace_root = tmp_path / "trace_v1"
    monkeypatch.setenv("EQNET_TRACE_V1", "1")
    monkeypatch.setenv("EQNET_TRACE_V1_DIR", str(trace_root))

    config = EQNetConfig(
        telemetry_dir=tmp_path / "telemetry",
        reports_dir=tmp_path / "reports",
        state_dir=tmp_path / "state",
        trace_dir=trace_root,
        audit_dir=tmp_path / "audit",
    )
    hub = EQNetHub(
        config=config,
        embed_text_fn=lambda text: [],
        idempotency_store=InMemoryIdempotencyStore(),
    )
    day = date(2025, 12, 14)
    hub.run_nightly(day, idempotency_key="nightly:test")
    hub.run_nightly(day, idempotency_key="nightly:test")

    day_dir = trace_root / "2025-12-14"
    files = list(day_dir.glob("hub-*.jsonl"))
    assert files
    records = [
        json.loads(line)
        for line in files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    statuses = [
        r.get("policy", {}).get("observations", {}).get("hub", {}).get("idempotency_status")
        for r in records
        if r.get("policy", {}).get("observations", {}).get("hub", {}).get("operation") == "run_nightly"
    ]
    assert "done" in statuses
    assert "skipped" in statuses


def test_run_nightly_shadow_records_delegate_exception_reason_code(tmp_path, monkeypatch):
    trace_root = tmp_path / "trace_v1"
    monkeypatch.setenv("EQNET_TRACE_V1", "1")
    monkeypatch.setenv("EQNET_TRACE_V1_DIR", str(trace_root))
    monkeypatch.setenv("HUB_DELEGATION_MODE", "shadow")
    monkeypatch.setenv("HUB_MISMATCH_POLICY", "fail")

    config = EQNetConfig(
        telemetry_dir=tmp_path / "telemetry",
        reports_dir=tmp_path / "reports",
        state_dir=tmp_path / "state",
        trace_dir=trace_root,
        audit_dir=tmp_path / "audit",
    )
    hub = EQNetHub(config=config, embed_text_fn=lambda text: [])

    class _FailingDelegate(_DelegateRuntime):
        def run_nightly(self, date_obj=None, *, idempotency_key=None):  # noqa: ANN001, ARG002
            super().run_nightly(date_obj, idempotency_key=idempotency_key)
            raise RuntimeError("nightly delegate boom")

    hub._runtime_delegate = _FailingDelegate()  # test-only override
    with pytest.raises(RuntimeError):
        hub.run_nightly(date(2025, 12, 14), idempotency_key="nightly:mismatch")
    day_dir = trace_root / "2025-12-14"
    files = list(day_dir.glob("hub-*.jsonl"))
    assert files
    records = [
        json.loads(line)
        for line in files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    nightly_records = [
        r
        for r in records
        if r.get("policy", {}).get("observations", {}).get("hub", {}).get("operation") == "run_nightly"
    ]
    assert nightly_records
    obs = nightly_records[-1]["policy"]["observations"]["hub"]
    assert obs.get("delegate_status") == "not_called"
    assert "DELEGATE_EXCEPTION" in (obs.get("mismatch_reason_codes") or [])


def test_run_nightly_on_mode_does_not_call_builtin_path(tmp_path, monkeypatch):
    trace_root = tmp_path / "trace_v1"
    monkeypatch.setenv("EQNET_TRACE_V1", "1")
    monkeypatch.setenv("EQNET_TRACE_V1_DIR", str(trace_root))
    monkeypatch.setenv("HUB_DELEGATION_MODE", "on")
    monkeypatch.setenv("HUB_MISMATCH_POLICY", "fail")

    delegate = _DelegateRuntime()
    config = EQNetConfig(
        telemetry_dir=tmp_path / "telemetry",
        reports_dir=tmp_path / "reports",
        state_dir=tmp_path / "state",
        trace_dir=trace_root,
        audit_dir=tmp_path / "audit",
    )
    hub = EQNetHub(config=config, embed_text_fn=lambda text: [], runtime_delegate=delegate)

    assert not hasattr(hub, "_builtin_runtime_delegate")
    hub.run_nightly(date(2025, 12, 14))
    assert delegate.calls
