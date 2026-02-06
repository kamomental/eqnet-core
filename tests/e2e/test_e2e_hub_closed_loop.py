from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from eqnet.hub.api import EQNetConfig, EQNetHub
from tests.e2e.proof_utils import load_proof_scenario, prepare_event


from types import SimpleNamespace

def _as_moment_object(event: dict) -> object:
    payload = dict(event)
    timestamp_ms = payload.get("timestamp_ms", 0)
    payload["timestamp"] = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
    return SimpleNamespace(**payload)


def _closed_loop_counts_from_trace(day_dir: Path) -> dict[str, int]:
    life_count = 0
    policy_count = 0
    output_count = 0
    linked_count = 0
    for fp in sorted(day_dir.glob("*.jsonl")):
        for line in fp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            policy_obs = (((row.get("policy") or {}).get("observations") or {}).get("hub") or {})
            qualia_obs = (((row.get("qualia") or {}).get("observations") or {}).get("hub") or {})
            day_key = policy_obs.get("day_key")
            if not isinstance(day_key, str) or not day_key.strip():
                continue
            life_fp = qualia_obs.get("life_indicator_fingerprint")
            policy_fp = qualia_obs.get("policy_prior_fingerprint")
            output_fp = qualia_obs.get("output_control_fingerprint")
            has_life = isinstance(life_fp, str) and bool(life_fp.strip())
            has_policy = isinstance(policy_fp, str) and bool(policy_fp.strip())
            has_output = isinstance(output_fp, str) and bool(output_fp.strip())
            if has_life:
                life_count += 1
            if has_policy:
                policy_count += 1
            if has_output:
                output_count += 1
            if has_life and has_policy and has_output:
                linked_count += 1
    return {
        "life_indicator_count": life_count,
        "policy_prior_count": policy_count,
        "output_control_count": output_count,
        "linked_count": linked_count,
    }


def _write_memory_reference_log_for_day(path: Path, day: date) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime(day.year, day.month, day.day, 12, 0, tzinfo=timezone.utc).timestamp()
    iso = day.isocalendar()
    payload = {
        "ts": stamp,
        "mode": "recall",
        "fidelity": 0.8,
        "reply_len": 24,
        "recall_render_mode": "cue_v1",
        "cue_label": "quiet walk",
        "rarity_budget": {
            "day_key": day.strftime("%Y-%m-%d"),
            "week_key": f"{iso.year}-W{iso.week:02d}",
            "daily_limit": 3,
            "weekly_limit": 12,
            "day_used": 1,
            "week_used": 1,
            "suppressed": False,
            "reason": "ok",
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_repair_event_sequence(hub: EQNetHub, *, scenario_id: str, day: date) -> None:
    base = datetime(day.year, day.month, day.day, 13, 0, tzinfo=timezone.utc)
    sequence = [
        ("TRIGGER", ["USER_DISTRESS"]),
        ("ACK", []),
        ("CALM", []),
        ("COMMIT", []),
    ]
    for idx, (event_name, reason_codes) in enumerate(sequence):
        ts = int((base + timedelta(seconds=idx)).timestamp() * 1000)
        payload: dict[str, Any] = {
            "timestamp_ms": ts,
            "scenario_id": scenario_id,
            "session_id": scenario_id,
            "turn_id": f"repair-{idx}",
            "user_text": "<redacted>",
            "repair_event": event_name,
        }
        if reason_codes:
            payload["reason_codes"] = reason_codes
        moment = _as_moment_object(payload)
        hub.log_moment(moment, "<redacted>")


def _repair_counts_from_trace(day_dir: Path) -> dict[str, int]:
    trigger_count = 0
    next_step_count = 0
    output_control_repair_state_count = 0
    for fp in sorted(day_dir.glob("*.jsonl")):
        for line in fp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            policy_obs = (((row.get("policy") or {}).get("observations") or {}).get("hub") or {})
            if policy_obs.get("repair_event") == "TRIGGER":
                trigger_count += 1
            if policy_obs.get("repair_state_after") == "NEXT_STEP":
                next_step_count += 1
            repair_state = policy_obs.get("output_control_repair_state")
            if isinstance(repair_state, str) and bool(repair_state.strip()):
                output_control_repair_state_count += 1
    return {
        "trigger_count": trigger_count,
        "next_step_count": next_step_count,
        "output_control_repair_state_count": output_control_repair_state_count,
    }


def _trace_rows(day_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fp in sorted(day_dir.glob("*.jsonl")):
        for line in fp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def test_hub_closed_loop_generates_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = load_proof_scenario()
    memory_ref_log = tmp_path / "logs" / "memory_ref.jsonl"
    config = EQNetConfig(
        telemetry_dir=tmp_path / "telemetry",
        reports_dir=tmp_path / "reports",
        state_dir=tmp_path / "state",
        trace_dir=tmp_path / "trace_v1",
        audit_dir=tmp_path / "audit",
        memory_reference_log_path=memory_ref_log,
    )
    hub = EQNetHub(config=config, embed_text_fn=lambda text: [])

    monkeypatch.setenv("EQNET_TRACE_V1", "1")
    monkeypatch.setenv("EQNET_TRACE_V1_DIR", str(config.trace_dir))

    for idx, raw in enumerate(scenario.events):
        event = prepare_event(raw, idx, scenario.scenario_id, scenario.date)
        moment = _as_moment_object(event)
        hub.log_moment(moment, event.get("user_text", "<redacted>"))

    _write_repair_event_sequence(hub, scenario_id=scenario.scenario_id, day=scenario.date)
    _write_memory_reference_log_for_day(memory_ref_log, scenario.date)
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
    closed_loop = audit.get("closed_loop_trace") or {}
    assert closed_loop.get("closed_loop_trace_ok") is True
    assert int(closed_loop.get("life_indicator_count") or 0) >= 1
    assert int(closed_loop.get("policy_prior_count") or 0) >= 1
    assert int(closed_loop.get("output_control_count") or 0) >= 1
    assert int(closed_loop.get("linked_count") or 0) >= 1

    counts = _closed_loop_counts_from_trace(day_dir)
    assert counts["life_indicator_count"] >= 1
    assert counts["policy_prior_count"] >= 1
    assert counts["output_control_count"] >= 1
    assert counts["linked_count"] >= 1
    recall = audit.get("recall_cue_budget") or {}
    assert recall.get("recall_cue_ok") is True
    assert recall.get("rarity_budget_ok") is True
    assert (recall.get("missing_keys") or []) == []
    recall_counts = recall.get("counts") or {}
    assert int(recall_counts.get("cue_v1_count") or 0) >= 1
    assert int(recall_counts.get("suppressed_count") or 0) >= 0
    repair = audit.get("repair_coverage") or {}
    assert int(repair.get("trigger_count") or 0) >= 1
    assert int(repair.get("progressed_count") or 0) >= 1
    assert int(repair.get("next_step_count") or 0) >= 1
    assert repair.get("stuck_suspected") is False

    repair_counts = _repair_counts_from_trace(day_dir)
    assert repair_counts["trigger_count"] >= 1
    assert repair_counts["next_step_count"] >= 1
    assert repair_counts["output_control_repair_state_count"] >= 1


def test_hub_entropy_memory_ops_e2e_sealed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    day = date(2025, 12, 14)
    memory_ref_log = tmp_path / "logs" / "memory_ref.jsonl"
    config = EQNetConfig(
        telemetry_dir=tmp_path / "telemetry",
        reports_dir=tmp_path / "reports",
        state_dir=tmp_path / "state",
        trace_dir=tmp_path / "trace_v1",
        audit_dir=tmp_path / "audit",
        memory_reference_log_path=memory_ref_log,
        memory_thermo_policy={
            "default_phase": "stabilization",
            "memory_phase_override": "exploration",
            "energy_budget_limit": 0.0,
        },
    )
    hub = EQNetHub(config=config, embed_text_fn=lambda text: [])
    monkeypatch.setenv("EQNET_TRACE_V1", "1")
    monkeypatch.setenv("EQNET_TRACE_V1_DIR", str(config.trace_dir))

    # Nightly #1: exploration phase
    hub.run_nightly(day)
    # Nightly #2: recovery phase (policy override)
    config.memory_thermo_policy = dict(config.memory_thermo_policy or {})
    config.memory_thermo_policy["memory_phase_override"] = "recovery"
    hub.run_nightly(day)

    # One conversation event to ensure gate reflection appears in hub trace.
    ts = datetime(day.year, day.month, day.day, 18, 0, tzinfo=timezone.utc)
    event = {
        "timestamp_ms": int(ts.timestamp() * 1000),
        "scenario_id": "entropy-e2e",
        "session_id": "entropy-e2e",
        "turn_id": "entropy-turn-1",
        "user_text": "<redacted>",
    }
    hub.log_moment(_as_moment_object(event), "<redacted>")

    day_key = day.strftime("%Y-%m-%d")
    audit_path = config.audit_dir / f"nightly_audit_{day_key}.json"
    assert audit_path.exists()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    thermo = audit.get("memory_thermo_contract") or {}
    assert thermo.get("memory_thermo_contract_ok") is True
    assert int(thermo.get("phase_transition_fp_stale_count") or 0) == 0

    day_dir = config.trace_dir / day_key
    rows = _trace_rows(day_dir)
    assert rows
    obs_rows = [
        (((r.get("policy") or {}).get("observations") or {}).get("hub") or {})
        for r in rows
    ]
    nightly_obs = [o for o in obs_rows if o.get("operation") == "run_nightly"]
    assert nightly_obs
    assert any(isinstance(o.get("defrag_metrics_before"), dict) for o in nightly_obs)
    assert any(isinstance(o.get("defrag_metrics_after"), dict) for o in nightly_obs)
    assert any(isinstance(o.get("defrag_metrics_delta"), dict) for o in nightly_obs)
    phases = [str(o.get("memory_phase") or "") for o in nightly_obs]
    assert "exploration" in phases
    assert "recovery" in phases
    projection_fps = [str(o.get("value_projection_fingerprint") or "") for o in nightly_obs if isinstance(o.get("value_projection_fingerprint"), str)]
    assert len(set(projection_fps)) >= 2
    assert any(bool(o.get("budget_throttle_applied")) for o in nightly_obs)
    assert any(str(o.get("throttle_reason_code") or "") == "BUDGET_EXCEEDED" for o in nightly_obs)
    assert any(str(o.get("output_control_profile") or "") == "cautious_budget_v1" for o in nightly_obs)

    log_obs = [o for o in obs_rows if not o.get("operation")]
    assert log_obs
    assert any(bool(o.get("budget_throttle_applied")) for o in log_obs)
    assert any(str(o.get("output_control_profile") or "") == "cautious_budget_v1" for o in log_obs)

