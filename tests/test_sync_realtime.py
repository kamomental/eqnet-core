from __future__ import annotations

from pathlib import Path

from eqnet.runtime.sync_realtime import (
    evaluate_downshift_state,
    evaluate_sync_micro_outcome,
    is_sync_emit_suppressed,
    load_realtime_downshift_policy,
)
from eqnet.runtime.future_contracts import load_sync_quality_rules


def test_evaluate_sync_micro_outcome_helped_and_unknown() -> None:
    rules = load_sync_quality_rules(Path("configs/sync_quality_rules_v0.yaml"))
    helped = evaluate_sync_micro_outcome(
        baseline_r=0.3,
        observed_r=0.35,
        window_sec=60,
        evaluated_at_eval_ts_ms=1000,
        rules=rules,
    )
    assert helped["result"] == "HELPED"
    unknown = evaluate_sync_micro_outcome(
        baseline_r=None,
        observed_r=0.35,
        window_sec=60,
        evaluated_at_eval_ts_ms=1000,
        rules=rules,
    )
    assert unknown["result"] == "UNKNOWN"


def test_evaluate_downshift_state_applies_on_harm_streak() -> None:
    policy = load_realtime_downshift_policy(Path("configs/realtime_downshift_policy_v0.yaml"))
    outcomes = [
        {"result": "NO_EFFECT", "evaluated_at_eval_ts_ms": 10},
        {"result": "HARMED", "evaluated_at_eval_ts_ms": 20},
        {"result": "HARMED", "evaluated_at_eval_ts_ms": 30},
    ]
    state = evaluate_downshift_state(outcomes=outcomes, now_ts_ms=1000, policy=policy)
    assert state["applied"] is True
    assert "DOWNSHIFT_HARM_STREAK" in state["reason_codes"]
    assert int(state["cooldown_until_ts_ms"]) > 1000


def test_is_sync_emit_suppressed_respects_cooldown() -> None:
    assert is_sync_emit_suppressed(now_ts_ms=1000, latest_downshift={"cooldown_until_ts_ms": 2000}) is True
    assert is_sync_emit_suppressed(now_ts_ms=3000, latest_downshift={"cooldown_until_ts_ms": 2000}) is False
