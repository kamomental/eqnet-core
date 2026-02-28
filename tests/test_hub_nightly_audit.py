from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

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
    assert isinstance(ru.get("tool_call_attempt_count"), int)
    assert isinstance(ru.get("tool_call_blocked_count"), int)
    assert isinstance(ru.get("forced_human_confirm_count"), int)
    assert isinstance(ru.get("online_delta_applied_count"), int)
    assert isinstance(ru.get("online_delta_missing_contract_count"), int)
    assert isinstance(ru.get("blocked_tool_names_topk"), list)
    assert isinstance(ru.get("blocked_reason_codes_topk"), list)
    assert isinstance(ru.get("forced_gate_action_topk"), list)
    assert isinstance((ru.get("online_delta_effectiveness") or {}).get("tool_block_rate"), (int, float))
    closed = payload.get("closed_loop_trace") or {}
    assert isinstance(closed.get("closed_loop_trace_ok"), bool)
    assert isinstance(closed.get("missing_keys"), list)
    recall = payload.get("recall_cue_budget") or {}
    assert isinstance(recall.get("recall_cue_ok"), bool)
    assert isinstance(recall.get("rarity_budget_ok"), bool)
    assert isinstance(recall.get("counts"), dict)
    repair = payload.get("repair_coverage") or {}
    assert isinstance(repair.get("repair_events_total"), int)
    assert isinstance(repair.get("trigger_count"), int)
    assert isinstance(repair.get("state_counts"), dict)
    assert isinstance(repair.get("progressed_count"), int)
    assert isinstance(repair.get("next_step_count"), int)
    assert isinstance(repair.get("stuck_suspected"), bool)
    assert isinstance(repair.get("missing_keys"), list)
    thermo = payload.get("memory_thermo_contract") or {}
    assert isinstance(thermo.get("memory_thermo_contract_ok"), bool)
    assert isinstance(thermo.get("missing_keys"), list)
    assert isinstance(thermo.get("events_checked"), int)
    assert isinstance(thermo.get("throttle_inconsistency_count"), int)
    assert isinstance(thermo.get("throttle_reason_missing_count"), int)
    assert isinstance(thermo.get("throttle_profile_missing_count"), int)
    assert isinstance(thermo.get("irreversible_without_trace_count"), int)
    assert isinstance(thermo.get("entropy_class_inconsistency_count"), int)
    assert isinstance(thermo.get("defrag_metrics_missing_count"), int)
    assert isinstance(thermo.get("defrag_delta_inconsistency_count"), int)
    assert isinstance(thermo.get("phase_transition_fp_stale_count"), int)
    assert isinstance(thermo.get("phase_override_applied_count"), int)
    assert isinstance(thermo.get("warnings"), list)
    immune_guard = payload.get("immune_guard") or {}
    assert isinstance(immune_guard.get("events_checked"), int)
    assert isinstance(immune_guard.get("quarantine_pruned_count"), int)
    assert isinstance(immune_guard.get("immune_guard_pruned_count"), int)
    assert isinstance(immune_guard.get("repeat_hit_rate"), (int, float))


def test_run_nightly_audit_closed_loop_ok_when_required_fingerprints_exist(tmp_path):
    trace_root = tmp_path / "trace_v1"
    day = "2025-12-14"
    day_dir = trace_root / day
    day_dir.mkdir(parents=True, exist_ok=True)
    sample = {
        "timestamp_ms": 1000,
        "turn_id": "turn-1",
        "scenario_id": "audit-demo",
        "source_loop": "hub",
        "policy": {
            "observations": {
                "hub": {
                    "operation": "run_nightly",
                    "day_key": day,
                }
            }
        },
        "qualia": {
            "observations": {
                "hub": {
                    "life_indicator_fingerprint": "life-fp",
                    "policy_prior_fingerprint": "policy-fp",
                    "output_control_fingerprint": "output-fp",
                }
            }
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
    payload = json.loads((tmp_path / "audit" / f"nightly_audit_{day}.json").read_text(encoding="utf-8"))
    closed = payload.get("closed_loop_trace") or {}
    assert closed.get("closed_loop_trace_ok") is True
    assert closed.get("missing_keys") == []


def test_run_nightly_audit_yellow_when_online_delta_block_rate_too_high(tmp_path):
    trace_root = tmp_path / "trace_v1"
    day = "2025-12-14"
    day_dir = trace_root / day
    day_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "timestamp_ms": 1000,
            "turn_id": "turn-1",
            "scenario_id": "audit-demo",
            "source_loop": "hub",
            "event_type": "tool_call",
            "ru_v0": {"gate_action": "EXECUTE", "policy_version": "ru-v0.1", "missing_required_fields": []},
        },
        {
            "timestamp_ms": 1001,
            "turn_id": "turn-2",
            "scenario_id": "audit-demo",
            "source_loop": "hub",
            "event_type": "tool_call_blocked",
            "tool_name": "web.fetch",
            "reason_codes": ["ONLINE_DELTA_TOOL_BLOCKED"],
            "ru_v0": {"gate_action": "HOLD", "policy_version": "ru-v0.1", "missing_required_fields": []},
        },
        {
            "timestamp_ms": 1002,
            "turn_id": "turn-3",
            "scenario_id": "audit-demo",
            "source_loop": "hub",
            "event_type": "tool_call_blocked",
            "tool_name": "web.fetch",
            "reason_codes": ["ONLINE_DELTA_TOOL_BLOCKED"],
            "ru_v0": {"gate_action": "HOLD", "policy_version": "ru-v0.1", "missing_required_fields": []},
        },
    ]
    (day_dir / "hub-1.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    config = EQNetConfig(
        telemetry_dir=tmp_path / "telemetry",
        reports_dir=tmp_path / "reports",
        state_dir=tmp_path / "state",
        trace_dir=trace_root,
        audit_dir=tmp_path / "audit",
        audit_thresholds={"online_delta_block_rate_yellow": 0.5},
    )
    hub = EQNetHub(config=config, embed_text_fn=lambda text: [])
    hub._run_nightly_audit(date(2025, 12, 14))
    payload = json.loads((tmp_path / "audit" / f"nightly_audit_{day}.json").read_text(encoding="utf-8"))
    health = payload.get("health") or {}
    reasons = [str(reason) for reason in (health.get("reasons") or [])]
    assert any("online delta tool block rate high" in reason for reason in reasons)


def test_run_nightly_audit_closed_loop_flags_missing_output_control(tmp_path):
    trace_root = tmp_path / "trace_v1"
    day = "2025-12-14"
    day_dir = trace_root / day
    day_dir.mkdir(parents=True, exist_ok=True)
    sample = {
        "timestamp_ms": 1000,
        "turn_id": "turn-1",
        "scenario_id": "audit-demo",
        "source_loop": "hub",
        "policy": {
            "observations": {
                "hub": {
                    "operation": "run_nightly",
                    "day_key": day,
                }
            }
        },
        "qualia": {
            "observations": {
                "hub": {
                    "life_indicator_fingerprint": "life-fp",
                    "policy_prior_fingerprint": "policy-fp",
                }
            }
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
    payload = json.loads((tmp_path / "audit" / f"nightly_audit_{day}.json").read_text(encoding="utf-8"))
    closed = payload.get("closed_loop_trace") or {}
    assert closed.get("closed_loop_trace_ok") is False
    assert "output_control_fingerprint" in (closed.get("missing_keys") or [])


def test_run_nightly_audit_includes_lazy_rag_sat_ratio_p95(tmp_path):
    trace_root = tmp_path / "trace_v1"
    day = "2025-12-14"
    day_dir = trace_root / day
    day_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "timestamp_ms": 1000,
            "turn_id": "turn-1",
            "scenario_id": "audit-demo",
            "source_loop": "hub",
            "response_meta": {"safety": {"lazy_rag_sat_ratio": 0.2}},
        },
        {
            "timestamp_ms": 1001,
            "turn_id": "turn-2",
            "scenario_id": "audit-demo",
            "source_loop": "hub",
            "response_meta": {"safety": {"lazy_rag_sat_ratio": 0.8}},
        },
    ]
    (day_dir / "hub-1.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    config = EQNetConfig(
        telemetry_dir=tmp_path / "telemetry",
        reports_dir=tmp_path / "reports",
        state_dir=tmp_path / "state",
        trace_dir=trace_root,
        audit_dir=tmp_path / "audit",
    )
    hub = EQNetHub(config=config, embed_text_fn=lambda text: [])
    hub._run_nightly_audit(date(2025, 12, 14))
    payload = json.loads((tmp_path / "audit" / f"nightly_audit_{day}.json").read_text(encoding="utf-8"))
    assert payload.get("lazy_rag_sat_ratio_count") == 2
    assert payload.get("lazy_rag_sat_ratio_p95") == 0.8


def test_run_nightly_audit_includes_uncertainty_reason_top3(tmp_path):
    trace_root = tmp_path / "trace_v1"
    day = "2025-12-14"
    day_dir = trace_root / day
    day_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "timestamp_ms": 1000,
            "turn_id": "turn-1",
            "response_meta": {
                "confidence": 0.40,
                "uncertainty_reason": ["retrieval_sparse", "score_saturation_high"],
            },
        },
        {
            "timestamp_ms": 1001,
            "turn_id": "turn-2",
            "response_meta": {
                "confidence": 0.62,
                "uncertainty_reason": ["retrieval_sparse"],
            },
        },
        {
            "timestamp_ms": 1002,
            "turn_id": "turn-3",
            "response_meta": {
                "confidence": 0.85,
                "uncertainty_reason": ["retrieval_error"],
            },
        },
    ]
    (day_dir / "hub-1.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    config = EQNetConfig(
        telemetry_dir=tmp_path / "telemetry",
        reports_dir=tmp_path / "reports",
        state_dir=tmp_path / "state",
        trace_dir=trace_root,
        audit_dir=tmp_path / "audit",
    )
    hub = EQNetHub(config=config, embed_text_fn=lambda text: [])
    hub._run_nightly_audit(date(2025, 12, 14))
    payload = json.loads((tmp_path / "audit" / f"nightly_audit_{day}.json").read_text(encoding="utf-8"))
    top3 = payload.get("uncertainty_reason_top3") or []
    assert top3
    assert top3[0]["reason"] == "retrieval_sparse"
    assert top3[0]["count"] == 2
    conf = payload.get("uncertainty_confidence") or {}
    assert conf.get("total") == 3
    assert conf.get("low") == 1
    assert conf.get("mid") == 1
    assert conf.get("high") == 1


def test_run_nightly_audit_mecpe_closed_loop_contract(tmp_path, monkeypatch):
    trace_root = tmp_path / "telemetry" / "trace_v1"
    monkeypatch.setenv("EQNET_TRACE_V1", "1")
    monkeypatch.setenv("EQNET_TRACE_V1_DIR", str(trace_root))

    config = EQNetConfig(
        telemetry_dir=tmp_path / "telemetry",
        reports_dir=tmp_path / "reports",
        state_dir=tmp_path / "state",
        trace_dir=trace_root,
        audit_dir=tmp_path / "audit",
    )
    hub = EQNetHub(config=config, embed_text_fn=lambda text: [])
    stamp = datetime(2025, 12, 14, 0, 0, tzinfo=timezone.utc)
    for idx in range(3):
        event = SimpleNamespace(
            timestamp=stamp,
            session_id="mecpe-e2e",
            turn_id=f"turn-{idx}",
            awareness_stage=1,
            emotion=SimpleNamespace(mask=0.1, love=0.2, stress=0.3, heart_rate_norm=0.4, breath_ratio_norm=0.5),
            culture=SimpleNamespace(rho=0.2, politeness=0.5, intimacy=0.4),
            mood={"arousal": 0.4, "stress": 0.2},
            metrics={"proximity": 0.7, "stress": 0.2},
            gate_context={"mode": "talk", "cultural_pressure": 0.1},
            talk_mode="talk",
            emotion_tag="neutral",
            audio_sha256="a" * 64,
            video_sha256="b" * 64,
        )
        hub.log_moment(
            event,
            f"user-text-{idx}",
        )

    hub._run_nightly_audit(date(2025, 12, 14))
    mecpe_files = sorted((tmp_path / "telemetry").glob("mecpe-*.jsonl"))
    assert len(mecpe_files) >= 1
    mecpe_rows = [
        line for line in mecpe_files[-1].read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(mecpe_rows) >= 1

    day = "2025-12-14"
    payload = json.loads((tmp_path / "audit" / f"nightly_audit_{day}.json").read_text(encoding="utf-8"))
    mecpe = payload.get("mecpe_audit") or {}
    assert mecpe
    assert int(mecpe.get("total_records") or 0) >= 1
    contract_errors = mecpe.get("contract_errors") or {}
    assert int(contract_errors.get("total") or 0) == 0
    hash_integrity = mecpe.get("hash_integrity") or {}
    assert float(hash_integrity.get("ok_rate") or 0.0) == 1.0


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
    assert obs.get("day_key") == "2025-12-14"
    assert obs.get("episode_id") == "nightly"
    assert obs.get("control_applied_at") == "nightly"
    assert obs.get("repair_state_before") == "RECOGNIZE"
    assert obs.get("repair_state_after") == "RECOGNIZE"
    assert obs.get("repair_event") == "NONE"
    assert isinstance(obs.get("repair_reason_codes"), list)
    assert isinstance(obs.get("repair_fingerprint"), str)
    assert obs.get("repair_fingerprint")
    qobs = nightly_records[-1]["qualia"]["observations"]["hub"]
    assert "life_indicator_fingerprint" in qobs
    assert "policy_prior_fingerprint" in qobs
    assert "output_control_fingerprint" in qobs
    assert "audit_fingerprint" in qobs
    assert isinstance(obs.get("memory_entropy_delta"), (int, float))
    assert isinstance(obs.get("memory_phase"), str)
    assert isinstance(obs.get("energy_budget_used"), (int, float))
    assert isinstance(obs.get("budget_throttle_applied"), bool)
    assert isinstance(obs.get("throttle_reason_code"), str)
    assert isinstance(obs.get("output_control_profile"), str)
    assert isinstance(obs.get("phase_override_applied"), bool)
    assert isinstance(obs.get("policy_version"), str)
    assert isinstance(obs.get("entropy_model_id"), str)
    assert isinstance(obs.get("event_id"), str)
    assert isinstance(obs.get("trace_id"), str)
    assert isinstance(obs.get("defrag_metrics_before"), dict)
    assert isinstance(obs.get("defrag_metrics_after"), dict)
    assert isinstance(obs.get("defrag_metrics_delta"), dict)


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


def test_run_nightly_audit_recall_cue_budget_ok(tmp_path):
    trace_root = tmp_path / "trace_v1"
    day = "2025-12-14"
    day_dir = trace_root / day
    day_dir.mkdir(parents=True, exist_ok=True)
    trace_sample = {
        "timestamp_ms": 1,
        "turn_id": "turn-1",
        "scenario_id": "audit-demo",
        "source_loop": "hub",
        "policy": {"observations": {"hub": {"day_key": day}}},
        "qualia": {"observations": {"hub": {"output_control_fingerprint": "fp"}}},
    }
    (day_dir / "hub-1.jsonl").write_text(json.dumps(trace_sample) + "\n", encoding="utf-8")
    memory_log = tmp_path / "logs" / "memory_ref.jsonl"
    memory_log.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime(2025, 12, 14, 12, 0, tzinfo=timezone.utc).timestamp()
    memory_sample = {
        "ts": ts,
        "recall_render_mode": "cue_v1",
        "rarity_budget": {
            "suppressed": False,
            "reason": "ok",
        },
    }
    memory_log.write_text(json.dumps(memory_sample) + "\n", encoding="utf-8")

    config = EQNetConfig(
        telemetry_dir=tmp_path / "telemetry",
        reports_dir=tmp_path / "reports",
        state_dir=tmp_path / "state",
        trace_dir=trace_root,
        audit_dir=tmp_path / "audit",
        memory_reference_log_path=memory_log,
    )
    hub = EQNetHub(config=config, embed_text_fn=lambda text: [])
    hub._run_nightly_audit(date(2025, 12, 14))
    payload = json.loads((tmp_path / "audit" / f"nightly_audit_{day}.json").read_text(encoding="utf-8"))
    recall = payload.get("recall_cue_budget") or {}
    assert recall.get("recall_cue_ok") is True
    assert recall.get("rarity_budget_ok") is True
    counts = recall.get("counts") or {}
    assert counts.get("memory_reference_total") == 1
    assert counts.get("cue_v1_count") == 1


def test_run_nightly_audit_recall_cue_budget_missing_fields(tmp_path):
    trace_root = tmp_path / "trace_v1"
    day = "2025-12-14"
    day_dir = trace_root / day
    day_dir.mkdir(parents=True, exist_ok=True)
    trace_sample = {
        "timestamp_ms": 1,
        "turn_id": "turn-1",
        "scenario_id": "audit-demo",
        "source_loop": "hub",
        "policy": {"observations": {"hub": {"day_key": day}}},
        "qualia": {"observations": {"hub": {"output_control_fingerprint": "fp"}}},
    }
    (day_dir / "hub-1.jsonl").write_text(json.dumps(trace_sample) + "\n", encoding="utf-8")
    memory_log = tmp_path / "logs" / "memory_ref.jsonl"
    memory_log.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime(2025, 12, 14, 12, 0, tzinfo=timezone.utc).timestamp()
    memory_sample = {
        "ts": ts,
        "mode": "recall",
    }
    memory_log.write_text(json.dumps(memory_sample) + "\n", encoding="utf-8")

    config = EQNetConfig(
        telemetry_dir=tmp_path / "telemetry",
        reports_dir=tmp_path / "reports",
        state_dir=tmp_path / "state",
        trace_dir=trace_root,
        audit_dir=tmp_path / "audit",
        memory_reference_log_path=memory_log,
    )
    hub = EQNetHub(config=config, embed_text_fn=lambda text: [])
    hub._run_nightly_audit(date(2025, 12, 14))
    payload = json.loads((tmp_path / "audit" / f"nightly_audit_{day}.json").read_text(encoding="utf-8"))
    recall = payload.get("recall_cue_budget") or {}
    assert recall.get("recall_cue_ok") is False
    assert recall.get("rarity_budget_ok") is False
    missing = recall.get("missing_keys") or []
    assert "recall_render_mode" in missing
    assert "rarity_budget" in missing


def test_run_nightly_audit_repair_coverage_progressed(tmp_path):
    trace_root = tmp_path / "trace_v1"
    day = "2025-12-14"
    day_dir = trace_root / day
    day_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "timestamp_ms": 1000,
            "turn_id": "turn-1",
            "source_loop": "hub",
            "policy": {"observations": {"hub": {
                "day_key": day,
                "repair_state_before": "RECOGNIZE",
                "repair_state_after": "RECOGNIZE",
                "repair_event": "TRIGGER",
                "repair_reason_codes": ["USER_DISTRESS"],
                "repair_fingerprint": "fp-1",
            }}},
            "qualia": {"observations": {"hub": {"output_control_fingerprint": "q"}}},
        },
        {
            "timestamp_ms": 2000,
            "turn_id": "turn-2",
            "source_loop": "hub",
            "policy": {"observations": {"hub": {
                "day_key": day,
                "repair_state_before": "RECOGNIZE",
                "repair_state_after": "NON_BLAME",
                "repair_event": "ACK",
                "repair_reason_codes": [],
                "repair_fingerprint": "fp-2",
            }}},
            "qualia": {"observations": {"hub": {"output_control_fingerprint": "q"}}},
        },
        {
            "timestamp_ms": 3000,
            "turn_id": "turn-3",
            "source_loop": "hub",
            "policy": {"observations": {"hub": {
                "day_key": day,
                "repair_state_before": "ACCEPT",
                "repair_state_after": "NEXT_STEP",
                "repair_event": "COMMIT",
                "repair_reason_codes": [],
                "repair_fingerprint": "fp-3",
            }}},
            "qualia": {"observations": {"hub": {"output_control_fingerprint": "q"}}},
        },
    ]
    (day_dir / "hub-1.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    config = EQNetConfig(
        telemetry_dir=tmp_path / "telemetry",
        reports_dir=tmp_path / "reports",
        state_dir=tmp_path / "state",
        trace_dir=trace_root,
        audit_dir=tmp_path / "audit",
    )
    hub = EQNetHub(config=config, embed_text_fn=lambda text: [])
    hub._run_nightly_audit(date(2025, 12, 14))
    payload = json.loads((tmp_path / "audit" / f"nightly_audit_{day}.json").read_text(encoding="utf-8"))
    repair = payload.get("repair_coverage") or {}
    assert repair.get("trigger_count") >= 1
    assert repair.get("next_step_count") >= 1
    assert repair.get("stuck_suspected") is False


def test_run_nightly_audit_repair_coverage_stuck_yellow(tmp_path):
    trace_root = tmp_path / "trace_v1"
    day = "2025-12-14"
    day_dir = trace_root / day
    day_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "timestamp_ms": 1000,
            "turn_id": "turn-1",
            "source_loop": "hub",
            "policy": {"observations": {"hub": {
                "day_key": day,
                "repair_state_before": "RECOGNIZE",
                "repair_state_after": "RECOGNIZE",
                "repair_event": "TRIGGER",
                "repair_reason_codes": ["USER_DISTRESS"],
                "repair_fingerprint": "fp-1",
            }}},
            "qualia": {"observations": {"hub": {"output_control_fingerprint": "q"}}},
        },
    ]
    (day_dir / "hub-1.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    config = EQNetConfig(
        telemetry_dir=tmp_path / "telemetry",
        reports_dir=tmp_path / "reports",
        state_dir=tmp_path / "state",
        trace_dir=trace_root,
        audit_dir=tmp_path / "audit",
    )
    hub = EQNetHub(config=config, embed_text_fn=lambda text: [])
    hub._run_nightly_audit(date(2025, 12, 14))
    payload = json.loads((tmp_path / "audit" / f"nightly_audit_{day}.json").read_text(encoding="utf-8"))
    repair = payload.get("repair_coverage") or {}
    assert repair.get("trigger_count") >= 1
    assert repair.get("next_step_count") == 0
    assert repair.get("progressed_count") == 0
    assert repair.get("stuck_suspected") is True
    health = payload.get("health") or {}
    assert health.get("status") in {"YELLOW", "RED"}


def test_run_nightly_audit_memory_thermo_contract_yellow_on_throttle_inconsistency(tmp_path):
    trace_root = tmp_path / "trace_v1"
    day = "2025-12-14"
    day_dir = trace_root / day
    day_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "timestamp_ms": 1000,
            "turn_id": "turn-1",
            "source_loop": "hub",
            "policy": {"observations": {"hub": {
                "day_key": day,
                "memory_entropy_delta": 0.1,
                "memory_phase": "stabilization",
                "energy_budget_used": 1.0,
                "energy_budget_limit": 2.0,
                "budget_throttle_applied": True,
                "policy_version": "memory-ops-v1",
                "entropy_model_id": "entropy-model-v1",
                "event_id": "evt-1",
                "trace_id": "tr-1",
                "defrag_metrics_before": {"memory_item_count": 2.0},
                "defrag_metrics_after": {"memory_item_count": 3.0},
                "defrag_metrics_delta": {"memory_item_count": 1.0},
                "throttle_reason_code": "BUDGET_EXCEEDED",
                "output_control_profile": "cautious_budget_v1",
            }}},
            "qualia": {"observations": {"hub": {"output_control_fingerprint": "q"}}},
        },
    ]
    (day_dir / "hub-1.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    config = EQNetConfig(
        telemetry_dir=tmp_path / "telemetry",
        reports_dir=tmp_path / "reports",
        state_dir=tmp_path / "state",
        trace_dir=trace_root,
        audit_dir=tmp_path / "audit",
    )
    hub = EQNetHub(config=config, embed_text_fn=lambda text: [])
    hub._run_nightly_audit(date(2025, 12, 14))
    payload = json.loads((tmp_path / "audit" / f"nightly_audit_{day}.json").read_text(encoding="utf-8"))
    thermo = payload.get("memory_thermo_contract") or {}
    assert thermo.get("memory_thermo_contract_ok") is False
    assert int(thermo.get("throttle_inconsistency_count") or 0) >= 1
    health = payload.get("health") or {}
    assert health.get("status") in {"YELLOW", "RED"}


def test_run_nightly_audit_memory_thermo_contract_yellow_on_irreversible_without_trace(tmp_path):
    trace_root = tmp_path / "trace_v1"
    day = "2025-12-14"
    day_dir = trace_root / day
    day_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "timestamp_ms": 1000,
            "turn_id": "turn-1",
            "source_loop": "hub",
            "policy": {"observations": {"hub": {
                "day_key": day,
                "memory_entropy_delta": 0.2,
                "memory_phase": "stabilization",
                "energy_budget_used": 2.0,
                "energy_budget_limit": 1.0,
                "budget_throttle_applied": True,
                "irreversible_op": True,
                "entropy_cost_class": "MID",
                "policy_version": "memory-ops-v1",
                "entropy_model_id": "entropy-model-v1",
                "defrag_metrics_before": {"memory_item_count": 2.0},
                "defrag_metrics_after": {"memory_item_count": 3.0},
                "defrag_metrics_delta": {"memory_item_count": 1.0},
                "throttle_reason_code": "BUDGET_EXCEEDED",
                "output_control_profile": "cautious_budget_v1",
            }}},
            "qualia": {"observations": {"hub": {"output_control_fingerprint": "q"}}},
        },
    ]
    (day_dir / "hub-1.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    config = EQNetConfig(
        telemetry_dir=tmp_path / "telemetry",
        reports_dir=tmp_path / "reports",
        state_dir=tmp_path / "state",
        trace_dir=trace_root,
        audit_dir=tmp_path / "audit",
    )
    hub = EQNetHub(config=config, embed_text_fn=lambda text: [])
    hub._run_nightly_audit(date(2025, 12, 14))
    payload = json.loads((tmp_path / "audit" / f"nightly_audit_{day}.json").read_text(encoding="utf-8"))
    thermo = payload.get("memory_thermo_contract") or {}
    assert thermo.get("memory_thermo_contract_ok") is False
    assert int(thermo.get("irreversible_without_trace_count") or 0) >= 1
    health = payload.get("health") or {}
    assert health.get("status") in {"YELLOW", "RED"}


def test_run_nightly_audit_memory_thermo_contract_yellow_on_defrag_delta_mismatch(tmp_path):
    trace_root = tmp_path / "trace_v1"
    day = "2025-12-14"
    day_dir = trace_root / day
    day_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "timestamp_ms": 1000,
            "turn_id": "turn-1",
            "source_loop": "hub",
            "policy": {"observations": {"hub": {
                "day_key": day,
                "memory_entropy_delta": 0.2,
                "memory_phase": "stabilization",
                "energy_budget_used": 2.0,
                "energy_budget_limit": 1.0,
                "budget_throttle_applied": True,
                "irreversible_op": False,
                "entropy_cost_class": "MID",
                "policy_version": "memory-ops-v1",
                "entropy_model_id": "entropy-model-v1",
                "event_id": "evt-1",
                "trace_id": "tr-1",
                "defrag_metrics_before": {"memory_item_count": 2.0},
                "defrag_metrics_after": {"memory_item_count": 3.0},
                "defrag_metrics_delta": {"memory_item_count": 0.5},
                "throttle_reason_code": "BUDGET_EXCEEDED",
                "output_control_profile": "cautious_budget_v1",
            }}},
            "qualia": {"observations": {"hub": {"output_control_fingerprint": "q"}}},
        },
    ]
    (day_dir / "hub-1.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    config = EQNetConfig(
        telemetry_dir=tmp_path / "telemetry",
        reports_dir=tmp_path / "reports",
        state_dir=tmp_path / "state",
        trace_dir=trace_root,
        audit_dir=tmp_path / "audit",
    )
    hub = EQNetHub(config=config, embed_text_fn=lambda text: [])
    hub._run_nightly_audit(date(2025, 12, 14))
    payload = json.loads((tmp_path / "audit" / f"nightly_audit_{day}.json").read_text(encoding="utf-8"))
    thermo = payload.get("memory_thermo_contract") or {}
    assert thermo.get("memory_thermo_contract_ok") is False
    assert int(thermo.get("defrag_delta_inconsistency_count") or 0) >= 1
    health = payload.get("health") or {}
    assert health.get("status") in {"YELLOW", "RED"}


def test_run_nightly_audit_memory_thermo_contract_yellow_on_phase_transition_fp_stale(tmp_path):
    trace_root = tmp_path / "trace_v1"
    day = "2025-12-14"
    day_dir = trace_root / day
    day_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "timestamp_ms": 1000,
            "turn_id": "turn-1",
            "source_loop": "hub",
            "policy": {"observations": {"hub": {
                "day_key": day,
                "memory_entropy_delta": 0.1,
                "memory_phase": "exploration",
                "phase_weight_profile": "phase.exploration.v1",
                "value_projection_fingerprint": "fp-same",
                "energy_budget_used": 1.0,
                "energy_budget_limit": 1.0,
                "budget_throttle_applied": True,
                "irreversible_op": False,
                "entropy_cost_class": "MID",
                "policy_version": "memory-ops-v1",
                "entropy_model_id": "entropy-model-v1",
                "event_id": "evt-1",
                "trace_id": "tr-1",
                "defrag_metrics_before": {"memory_item_count": 2.0},
                "defrag_metrics_after": {"memory_item_count": 2.0},
                "defrag_metrics_delta": {"memory_item_count": 0.0},
                "throttle_reason_code": "BUDGET_EXCEEDED",
                "output_control_profile": "cautious_budget_v1",
            }}},
            "qualia": {"observations": {"hub": {"output_control_fingerprint": "q"}}},
        },
        {
            "timestamp_ms": 2000,
            "turn_id": "turn-2",
            "source_loop": "hub",
            "policy": {"observations": {"hub": {
                "day_key": day,
                "memory_entropy_delta": 0.2,
                "memory_phase": "recovery",
                "phase_weight_profile": "phase.recovery.v1",
                "value_projection_fingerprint": "fp-same",
                "energy_budget_used": 1.0,
                "energy_budget_limit": 1.0,
                "budget_throttle_applied": True,
                "irreversible_op": False,
                "entropy_cost_class": "MID",
                "policy_version": "memory-ops-v1",
                "entropy_model_id": "entropy-model-v1",
                "event_id": "evt-2",
                "trace_id": "tr-2",
                "defrag_metrics_before": {"memory_item_count": 2.0},
                "defrag_metrics_after": {"memory_item_count": 2.0},
                "defrag_metrics_delta": {"memory_item_count": 0.0},
                "throttle_reason_code": "BUDGET_EXCEEDED",
                "output_control_profile": "cautious_budget_v1",
            }}},
            "qualia": {"observations": {"hub": {"output_control_fingerprint": "q"}}},
        },
    ]
    (day_dir / "hub-1.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    config = EQNetConfig(
        telemetry_dir=tmp_path / "telemetry",
        reports_dir=tmp_path / "reports",
        state_dir=tmp_path / "state",
        trace_dir=trace_root,
        audit_dir=tmp_path / "audit",
    )
    hub = EQNetHub(config=config, embed_text_fn=lambda text: [])
    hub._run_nightly_audit(date(2025, 12, 14))
    payload = json.loads((tmp_path / "audit" / f"nightly_audit_{day}.json").read_text(encoding="utf-8"))
    thermo = payload.get("memory_thermo_contract") or {}
    assert thermo.get("memory_thermo_contract_ok") is False
    assert int(thermo.get("phase_transition_fp_stale_count") or 0) >= 1
    health = payload.get("health") or {}
    assert health.get("status") in {"YELLOW", "RED"}


def test_run_nightly_audit_memory_thermo_contract_yellow_on_missing_throttle_reason(tmp_path):
    trace_root = tmp_path / "trace_v1"
    day = "2025-12-14"
    day_dir = trace_root / day
    day_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_ms": 1000,
        "turn_id": "turn-1",
        "source_loop": "hub",
        "policy": {"observations": {"hub": {
            "day_key": day,
            "memory_entropy_delta": 0.2,
            "memory_phase": "stabilization",
            "phase_weight_profile": "phase.stabilization.v1",
            "value_projection_fingerprint": "fp-1",
            "energy_budget_used": 2.0,
            "energy_budget_limit": 1.0,
            "budget_throttle_applied": True,
            "output_control_profile": "cautious_budget_v1",
            "irreversible_op": False,
            "entropy_cost_class": "MID",
            "policy_version": "memory-ops-v1",
            "entropy_model_id": "entropy-model-v1",
            "event_id": "evt-1",
            "trace_id": "tr-1",
            "defrag_metrics_before": {"memory_item_count": 2.0},
            "defrag_metrics_after": {"memory_item_count": 2.0},
            "defrag_metrics_delta": {"memory_item_count": 0.0},
        }}},
        "qualia": {"observations": {"hub": {"output_control_fingerprint": "q"}}},
    }
    (day_dir / "hub-1.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    config = EQNetConfig(
        telemetry_dir=tmp_path / "telemetry",
        reports_dir=tmp_path / "reports",
        state_dir=tmp_path / "state",
        trace_dir=trace_root,
        audit_dir=tmp_path / "audit",
    )
    hub = EQNetHub(config=config, embed_text_fn=lambda text: [])
    hub._run_nightly_audit(date(2025, 12, 14))
    payload = json.loads((tmp_path / "audit" / f"nightly_audit_{day}.json").read_text(encoding="utf-8"))
    thermo = payload.get("memory_thermo_contract") or {}
    assert thermo.get("memory_thermo_contract_ok") is False
    assert int(thermo.get("throttle_reason_missing_count") or 0) >= 1


def test_run_nightly_audit_memory_thermo_contract_yellow_on_missing_throttle_profile(tmp_path):
    trace_root = tmp_path / "trace_v1"
    day = "2025-12-14"
    day_dir = trace_root / day
    day_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_ms": 1000,
        "turn_id": "turn-1",
        "source_loop": "hub",
        "policy": {"observations": {"hub": {
            "day_key": day,
            "memory_entropy_delta": 0.2,
            "memory_phase": "stabilization",
            "phase_weight_profile": "phase.stabilization.v1",
            "value_projection_fingerprint": "fp-1",
            "energy_budget_used": 2.0,
            "energy_budget_limit": 1.0,
            "budget_throttle_applied": True,
            "throttle_reason_code": "BUDGET_EXCEEDED",
            "irreversible_op": False,
            "entropy_cost_class": "MID",
            "policy_version": "memory-ops-v1",
            "entropy_model_id": "entropy-model-v1",
            "event_id": "evt-1",
            "trace_id": "tr-1",
            "defrag_metrics_before": {"memory_item_count": 2.0},
            "defrag_metrics_after": {"memory_item_count": 2.0},
            "defrag_metrics_delta": {"memory_item_count": 0.0},
        }}},
        "qualia": {"observations": {"hub": {"output_control_fingerprint": "q"}}},
    }
    (day_dir / "hub-1.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    config = EQNetConfig(
        telemetry_dir=tmp_path / "telemetry",
        reports_dir=tmp_path / "reports",
        state_dir=tmp_path / "state",
        trace_dir=trace_root,
        audit_dir=tmp_path / "audit",
    )
    hub = EQNetHub(config=config, embed_text_fn=lambda text: [])
    hub._run_nightly_audit(date(2025, 12, 14))
    payload = json.loads((tmp_path / "audit" / f"nightly_audit_{day}.json").read_text(encoding="utf-8"))
    thermo = payload.get("memory_thermo_contract") or {}
    assert thermo.get("memory_thermo_contract_ok") is False
    assert int(thermo.get("throttle_profile_missing_count") or 0) >= 1


def test_run_nightly_audit_memory_thermo_contract_warn_on_phase_override_applied(tmp_path):
    trace_root = tmp_path / "trace_v1"
    day = "2025-12-14"
    day_dir = trace_root / day
    day_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_ms": 1000,
        "turn_id": "turn-1",
        "source_loop": "hub",
        "policy": {"observations": {"hub": {
            "day_key": day,
            "memory_entropy_delta": 0.0,
            "memory_phase": "exploration",
            "phase_weight_profile": "phase.exploration.v1",
            "value_projection_fingerprint": "fp-1",
            "energy_budget_used": 0.0,
            "energy_budget_limit": 1.0,
            "budget_throttle_applied": False,
            "throttle_reason_code": "",
            "output_control_profile": "normal_v1",
            "phase_override_applied": True,
            "irreversible_op": False,
            "entropy_cost_class": "LOW",
            "policy_version": "memory-ops-v1",
            "entropy_model_id": "entropy-model-v1",
            "event_id": "evt-1",
            "trace_id": "tr-1",
            "defrag_metrics_before": {"memory_item_count": 1.0},
            "defrag_metrics_after": {"memory_item_count": 1.0},
            "defrag_metrics_delta": {"memory_item_count": 0.0},
        }}},
        "qualia": {"observations": {"hub": {"output_control_fingerprint": "q"}}},
    }
    (day_dir / "hub-1.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    config = EQNetConfig(
        telemetry_dir=tmp_path / "telemetry",
        reports_dir=tmp_path / "reports",
        state_dir=tmp_path / "state",
        trace_dir=trace_root,
        audit_dir=tmp_path / "audit",
    )
    hub = EQNetHub(config=config, embed_text_fn=lambda text: [])
    hub._run_nightly_audit(date(2025, 12, 14))
    payload = json.loads((tmp_path / "audit" / f"nightly_audit_{day}.json").read_text(encoding="utf-8"))
    thermo = payload.get("memory_thermo_contract") or {}
    assert thermo.get("memory_thermo_contract_ok") is True
    assert int(thermo.get("phase_override_applied_count") or 0) >= 1
    assert "PHASE_OVERRIDE_APPLIED" in (thermo.get("warnings") or [])
