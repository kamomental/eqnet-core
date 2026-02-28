from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from eqnet.hub.api import EQNetConfig, EQNetHub
from eqnet.runtime.external_runtime import ExternalRuntimeDelegateV2
from eqnet.runtime.nightly.metabolism_tool import (
    DEFAULT_METABOLISM_POLICY,
    run_metabolism_cycle,
)
from eqnet.runtime.nightly.repair_tool import run_repair_cycle


def test_run_metabolism_cycle_exposes_budget_fields() -> None:
    records = [
        {
            "timestamp": "2025-12-14T00:00:00+00:00",
            "qualia_vec": [0.1, 0.2, 0.3],
            "parent_id": "p1",
        }
    ]
    result = run_metabolism_cycle(
        qualia_records=records,
        policy=DEFAULT_METABOLISM_POLICY,
        previous_state=None,
    )
    assert result.get("metabolism_status") == "applied"
    assert result.get("metabolism_tool_version") == "metabolism_tool_v1"
    assert isinstance(result.get("resource_budget"), dict)
    assert isinstance(result.get("attention_budget_level"), float)
    assert isinstance(result.get("attention_budget_used"), float)
    assert isinstance(result.get("affect_budget_level"), float)
    assert isinstance(result.get("affect_budget_used"), float)
    assert isinstance(result.get("memory_entropy_delta"), float)
    assert isinstance(result.get("energy_budget_used"), float)
    assert result.get("metabolism_invariants_ok") is True
    assert isinstance(result.get("metabolism_conservation_error"), float)


def test_run_repair_cycle_rewrites_replay_memory_and_reports(tmp_path: Path) -> None:
    replay_path = tmp_path / "state" / "replay_memory.jsonl"
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_text(
        json.dumps(
            {
                "trace_id": "trace-1",
                "episode_id": "ep-1",
                "memory_kind": "episode",
                "weight": 0.5,
                "emotion_modulation": 0.0,
                "meta": {"replay": {"seeds": []}},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    config = SimpleNamespace(
        state_dir=tmp_path / "state",
        runtime_policy={
            "forgetting": {
                "enable": True,
                "base_delta": 0.1,
                "max_delta_w": 1.0,
                "min_w": 0.0,
                "max_w": 1.0,
                "recall_k": 0.0,
                "recall_weight": 0.0,
                "affect_weight": 0.0,
                "interference_weight": 0.0,
                "interference_k": 0.0,
                "reconsolidation_rate": 0.0,
                "monument_w_lock": True,
            }
        },
    )
    report = run_repair_cycle(config)
    assert isinstance(report, dict)
    assert report.get("repair_status") == "applied"
    assert report.get("repair_tool_version") == "repair_tool_v1"
    assert int(report.get("repaired_events_count") or 0) >= 1
    assert bool(report.get("rewrite_applied")) is True
    assert isinstance(report.get("repair_plan_id"), str)
    assert report.get("repair_plan_id")
    assert isinstance(report.get("replay_token"), str)
    ops = report.get("repair_ops") or []
    assert isinstance(ops, list)
    assert ops
    assert isinstance(report.get("repair_ops_digest"), str)
    assert report.get("repair_ops_digest")
    first_op = ops[0]
    assert isinstance(first_op.get("target_hash"), str)
    assert first_op.get("target_hash")
    assert "trace-1" not in json.dumps(first_op, ensure_ascii=False)

    rows = [
        json.loads(line)
        for line in replay_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    assert float(rows[0].get("weight") or 0.0) > 0.5


def test_run_nightly_trace_contains_metabolism_and_repair_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    day = date(2025, 12, 14)
    config = EQNetConfig(
        telemetry_dir=tmp_path / "telemetry",
        reports_dir=tmp_path / "reports",
        state_dir=tmp_path / "state",
        trace_dir=tmp_path / "trace_v1",
        audit_dir=tmp_path / "audit",
        runtime_policy={"forgetting": {"enable": False}},
    )
    hub = EQNetHub(config=config, embed_text_fn=lambda text: [])
    monkeypatch.setenv("EQNET_TRACE_V1", "1")
    monkeypatch.setenv("EQNET_TRACE_V1_DIR", str(config.trace_dir))

    qualia_path = config.telemetry_dir / "qualia-20251214.jsonl"
    qualia_path.parent.mkdir(parents=True, exist_ok=True)
    qualia_path.write_text(
        json.dumps(
            {
                "timestamp": "2025-12-14T08:00:00+00:00",
                "qualia_vec": [0.1, 0.2, 0.3],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    hub.run_nightly(day)

    day_dir = config.trace_dir / "2025-12-14"
    files = list(day_dir.glob("hub-*.jsonl"))
    assert files
    rows = [
        json.loads(line)
        for line in files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    nightly_obs = [
        (((row.get("policy") or {}).get("observations") or {}).get("hub") or {})
        for row in rows
        if (((row.get("policy") or {}).get("observations") or {}).get("hub") or {}).get("operation") == "run_nightly"
    ]
    assert nightly_obs
    obs = nightly_obs[-1]
    assert obs.get("metabolism_status") == "applied"
    assert obs.get("metabolism_tool_version") == "metabolism_tool_v1"
    assert obs.get("repair_status") == "disabled"
    assert obs.get("repair_tool_version") == "repair_tool_v1"
    assert isinstance(obs.get("repaired_events_count"), int)
    assert isinstance(obs.get("attention_budget_level"), (int, float))
    assert isinstance(obs.get("attention_budget_used"), (int, float))
    assert isinstance(obs.get("affect_budget_level"), (int, float))
    assert isinstance(obs.get("affect_budget_used"), (int, float))
    assert obs.get("metabolism_invariants_ok") is True
    assert isinstance(obs.get("metabolism_conservation_error"), (int, float))
    assert isinstance(obs.get("nightly_transaction_id"), str)
    assert obs.get("nightly_transaction_id")
    assert obs.get("nightly_transaction_phase") == "COMMITTED"
    assert obs.get("nightly_transaction_atomic") is True
    assert isinstance(obs.get("quarantine_pruned_count"), int)
    assert isinstance(obs.get("immune_guard_pruned_count"), int)
    assert isinstance(obs.get("repeat_hit_rate"), (int, float))


def test_run_nightly_atomic_commit_does_not_publish_partial_latest_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    day = date(2025, 12, 14)
    config = EQNetConfig(
        telemetry_dir=tmp_path / "telemetry",
        reports_dir=tmp_path / "reports",
        state_dir=tmp_path / "state",
        trace_dir=tmp_path / "trace_v1",
        audit_dir=tmp_path / "audit",
        runtime_policy={"forgetting": {"enable": False}},
    )
    hub = EQNetHub(config=config, embed_text_fn=lambda text: [])
    delegate = hub._runtime_delegate
    assert isinstance(delegate, ExternalRuntimeDelegateV2)

    qualia_path = config.telemetry_dir / "qualia-20251214.jsonl"
    qualia_path.parent.mkdir(parents=True, exist_ok=True)
    qualia_path.write_text(
        json.dumps(
            {
                "timestamp": "2025-12-14T08:00:00+00:00",
                "qualia_vec": [0.1, 0.2, 0.3],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    def _boom_commit(_staged, *, tx_id=None):
        raise RuntimeError("commit failed")

    monkeypatch.setattr(delegate, "_commit_nightly_artifacts", _boom_commit)
    before = hub._get_latest_state()
    assert not before.get("memory_thermo")

    try:
        hub.run_nightly(day)
        assert False, "run_nightly should fail when commit fails"
    except RuntimeError:
        pass

    after = hub._get_latest_state()
    assert not after.get("memory_thermo")
    assert after.get("life_indicator") is None
    assert after.get("policy_prior") is None
    assert not (config.state_dir / "memory-thermo-latest.json").exists()


def test_run_nightly_same_idempotency_key_skips_second_commit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    day = date(2025, 12, 14)
    config = EQNetConfig(
        telemetry_dir=tmp_path / "telemetry",
        reports_dir=tmp_path / "reports",
        state_dir=tmp_path / "state",
        trace_dir=tmp_path / "trace_v1",
        audit_dir=tmp_path / "audit",
        runtime_policy={"forgetting": {"enable": False}},
    )
    hub = EQNetHub(config=config, embed_text_fn=lambda text: [])
    delegate = hub._runtime_delegate
    assert isinstance(delegate, ExternalRuntimeDelegateV2)

    qualia_path = config.telemetry_dir / "qualia-20251214.jsonl"
    qualia_path.parent.mkdir(parents=True, exist_ok=True)
    qualia_path.write_text(
        json.dumps(
            {
                "timestamp": "2025-12-14T08:00:00+00:00",
                "qualia_vec": [0.1, 0.2, 0.3],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    commit_calls = {"count": 0}
    orig_commit = delegate._commit_nightly_artifacts

    def _count_commit(staged, *, tx_id=None):
        commit_calls["count"] += 1
        return orig_commit(staged, tx_id=tx_id)

    monkeypatch.setattr(delegate, "_commit_nightly_artifacts", _count_commit)
    monkeypatch.setenv("EQNET_NIGHTLY_TX_IDEMPOTENT", "1")
    hub.run_nightly(day, idempotency_key="fixed-nightly-key")
    hub.run_nightly(day, idempotency_key="fixed-nightly-key")

    assert commit_calls["count"] == 1
