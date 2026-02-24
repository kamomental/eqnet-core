from __future__ import annotations

from pathlib import Path

from eqnet.runtime.replay.diff_gate import evaluate_diff_gate, load_diff_gate_policy


def test_replay_diff_gate_passes_with_required_fields() -> None:
    policy = load_diff_gate_policy(Path("configs/diff_gate_policy_v0.yaml"))
    payload = {
        "comparison_scope": {
            "trace_path": "assets/replay/trace.jsonl",
            "start_day_key": "2026-02-01",
            "end_day_key": "2026-02-07",
            "config_set_a": "A",
            "config_set_b": "B",
            "eval_ts_ms": 1767139140000,
        },
        "config_set_a_meta": {"files": {"fsm_policy_v0.yaml": {"fingerprint": "aaa"}}},
        "config_set_b_meta": {"files": {"fsm_policy_v0.yaml": {"fingerprint": "bbb"}}},
        "top_regressions": [{"metric": "preventive_harmed_rate", "delta": 0.1}],
    }
    result = evaluate_diff_gate(payload, policy)
    assert bool(result.get("ok"))
    assert str(result.get("schema_version") or "") == "gate_result_v0"
    assert int(result.get("exit_code", -1)) == 0
    assert str(result.get("severity") or "") == "OK"
    assert not result.get("errors")
    assert not (result.get("reason_codes") or [])
    assert str(result.get("gate_policy_fingerprint") or "")
    assert str(result.get("diff_summary_fingerprint") or "")
    assert int(result.get("evaluated_at_eval_ts_ms") or 0) == 1767139140000
    assert (result.get("comparison_scope") or {}).get("config_set_b") == "B"


def test_replay_diff_gate_fails_when_scope_or_priority_missing() -> None:
    policy = load_diff_gate_policy(Path("configs/diff_gate_policy_v0.yaml"))
    payload = {
        "comparison_scope": {"start_day_key": "2026-02-01"},
        "config_set_a_meta": {"files": {"fsm_policy_v0.yaml": {"fingerprint": "aaa"}}},
        "config_set_b_meta": {"files": {"fsm_policy_v0.yaml": {"fingerprint": "bbb"}}},
        "top_regressions": [{"metric": "preventive_helped_rate", "delta": 0.1}],
    }
    result = evaluate_diff_gate(payload, policy)
    assert not bool(result.get("ok"))
    assert int(result.get("exit_code", 0)) == 2
    assert str(result.get("severity") or "") == "BLOCKER"
    errors = set(result.get("errors") or [])
    assert "comparison_scope_required" in errors
    assert "top_regression_priority" in errors
    reason_codes = set(result.get("reason_codes") or [])
    assert "GATE_MISSING_SCOPE_KEY" in reason_codes
    assert "GATE_TOP_REGRESSION_NOT_RISK_METRIC" in reason_codes


def test_replay_diff_gate_accepts_sync_risk_metric_as_top_regression() -> None:
    policy = load_diff_gate_policy(Path("configs/diff_gate_policy_v0.yaml"))
    payload = {
        "comparison_scope": {
            "trace_path": "assets/replay/trace.jsonl",
            "start_day_key": "2026-02-01",
            "end_day_key": "2026-02-07",
            "config_set_a": "A",
            "config_set_b": "B",
        },
        "config_set_a_meta": {"files": {"fsm_policy_v0.yaml": {"fingerprint": "aaa"}}},
        "config_set_b_meta": {"files": {"fsm_policy_v0.yaml": {"fingerprint": "bbb"}}},
        "top_regressions": [{"metric": "sync_outcome_harmed_rate", "delta": 0.1}],
    }
    result = evaluate_diff_gate(payload, policy)
    assert bool(result.get("ok"))
    assert int(result.get("exit_code", -1)) == 0


def test_replay_diff_gate_rejects_improvement_metric_as_top_regression() -> None:
    policy = load_diff_gate_policy(Path("configs/diff_gate_policy_v0.yaml"))
    payload = {
        "comparison_scope": {
            "trace_path": "assets/replay/trace.jsonl",
            "start_day_key": "2026-02-01",
            "end_day_key": "2026-02-07",
            "config_set_a": "A",
            "config_set_b": "B",
        },
        "config_set_a_meta": {"files": {"fsm_policy_v0.yaml": {"fingerprint": "aaa"}}},
        "config_set_b_meta": {"files": {"fsm_policy_v0.yaml": {"fingerprint": "bbb"}}},
        "top_regressions": [{"metric": "sync_r_median_avg", "delta": 0.1}],
    }
    result = evaluate_diff_gate(payload, policy)
    assert not bool(result.get("ok"))
    assert int(result.get("exit_code", 0)) == 2
    reason_codes = set(result.get("reason_codes") or [])
    assert "GATE_TOP_REGRESSION_IS_IMPROVEMENT_METRIC" in reason_codes


def test_replay_diff_gate_blocks_green_regression_metric_in_aggregate_delta() -> None:
    policy = load_diff_gate_policy(Path("configs/diff_gate_policy_v0.yaml"))
    payload = {
        "comparison_scope": {
            "trace_path": "assets/replay/trace.jsonl",
            "start_day_key": "2026-02-01",
            "end_day_key": "2026-02-07",
            "config_set_a": "A",
            "config_set_b": "B",
        },
        "config_set_a_meta": {"files": {"fsm_policy_v0.yaml": {"fingerprint": "aaa"}}},
        "config_set_b_meta": {"files": {"fsm_policy_v0.yaml": {"fingerprint": "bbb"}}},
        "top_regressions": [{"metric": "sync_outcome_harmed_rate", "delta": 0.1}],
        "aggregate_delta": {
            "sync_micro_harmed_rate": 0.01,
            "sync_downshift_applied_rate": 0.0,
            "sync_micro_unknown_rate": 0.0,
        },
    }
    result = evaluate_diff_gate(payload, policy)
    assert not bool(result.get("ok"))
    assert int(result.get("exit_code", 0)) == 2
    reason_codes = set(result.get("reason_codes") or [])
    assert "GREEN_REGRESSION_SYNC_MICRO_HARMED_RATE" in reason_codes


def test_replay_diff_gate_passes_when_green_regression_metrics_do_not_worsen() -> None:
    policy = load_diff_gate_policy(Path("configs/diff_gate_policy_v0.yaml"))
    payload = {
        "comparison_scope": {
            "trace_path": "assets/replay/trace.jsonl",
            "start_day_key": "2026-02-01",
            "end_day_key": "2026-02-07",
            "config_set_a": "A",
            "config_set_b": "B",
        },
        "config_set_a_meta": {"files": {"fsm_policy_v0.yaml": {"fingerprint": "aaa"}}},
        "config_set_b_meta": {"files": {"fsm_policy_v0.yaml": {"fingerprint": "bbb"}}},
        "top_regressions": [{"metric": "sync_outcome_harmed_rate", "delta": 0.1}],
        "aggregate_delta": {
            "sync_micro_harmed_rate": 0.0,
            "sync_downshift_applied_rate": -0.02,
            "sync_micro_unknown_rate": 0.0,
        },
    }
    result = evaluate_diff_gate(payload, policy)
    assert bool(result.get("ok"))
    assert int(result.get("exit_code", -1)) == 0


def test_replay_diff_gate_blocks_behavior_change_regression_metric_in_aggregate_delta() -> None:
    policy = load_diff_gate_policy(Path("configs/diff_gate_policy_v0.yaml"))
    payload = {
        "comparison_scope": {
            "trace_path": "assets/replay/trace.jsonl",
            "start_day_key": "2026-02-01",
            "end_day_key": "2026-02-07",
            "config_set_a": "A",
            "config_set_b": "B",
        },
        "config_set_a_meta": {"files": {"fsm_policy_v0.yaml": {"fingerprint": "aaa"}}},
        "config_set_b_meta": {"files": {"fsm_policy_v0.yaml": {"fingerprint": "bbb"}}},
        "top_regressions": [{"metric": "sync_outcome_harmed_rate", "delta": 0.1}],
        "aggregate_delta": {
            "behavior_change_harmed_rate_delta_avg": 0.05,
            "behavior_change_reject_rate_delta_avg": 0.0,
        },
    }
    result = evaluate_diff_gate(payload, policy)
    assert not bool(result.get("ok"))
    assert int(result.get("exit_code", 0)) == 2
    reason_codes = set(result.get("reason_codes") or [])
    assert "BC_REGRESSION_HARMED_RATE_DELTA" in reason_codes
