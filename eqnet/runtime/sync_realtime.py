from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

from eqnet.runtime.future_contracts import load_yaml_with_default


def load_realtime_downshift_policy(path: Path) -> Dict[str, Any]:
    default = {
        "schema_version": "realtime_downshift_policy_v0",
        "policy_version": "realtime_downshift_policy_v0",
        "policy_source": "configs/realtime_downshift_policy_v0.yaml",
        "harm_streak_threshold": 2,
        "unknown_streak_threshold": 3,
        "cooldown_sec": 300,
        "actions": [
            "STOP_SYNC_PROPOSALS",
            "LOWER_COUPLING_CAP",
            "INCREASE_MIN_CONFIDENCE_TO_EMIT",
        ],
        "reason_codes": {
            "harm_streak": "DOWNSHIFT_HARM_STREAK",
            "unknown_streak": "DOWNSHIFT_UNKNOWN_STREAK",
            "active": "DOWNSHIFT_ACTIVE",
        },
    }
    return load_yaml_with_default(path, default)


def evaluate_sync_micro_outcome(
    *,
    baseline_r: float | None,
    observed_r: float | None,
    window_sec: int,
    evaluated_at_eval_ts_ms: int,
    rules: Mapping[str, Any],
) -> Dict[str, Any]:
    rc = (rules.get("reason_codes") or {}) if isinstance(rules, Mapping) else {}
    th = (rules.get("outcome_thresholds") or {}) if isinstance(rules, Mapping) else {}
    helped_delta = float(th.get("r_helped_delta", 0.02) or 0.02)
    harmed_delta = float(th.get("r_harmed_delta", -0.02) or -0.02)
    if not isinstance(baseline_r, (int, float)) or not isinstance(observed_r, (int, float)):
        return {
            "result": "UNKNOWN",
            "reason_codes": [str(rc.get("sync_unknown_missing_observed") or "SYNC_UNKNOWN_MISSING_OBSERVED")],
            "delta_r": None,
            "window_sec": int(window_sec),
            "evaluated_at_eval_ts_ms": int(evaluated_at_eval_ts_ms),
        }
    delta = float(observed_r) - float(baseline_r)
    if delta >= helped_delta:
        result = "HELPED"
        reasons = [str(rc.get("sync_helped_r_up") or "SYNC_HELPED_R_UP")]
    elif delta <= harmed_delta:
        result = "HARMED"
        reasons = [str(rc.get("sync_harmed_r_down") or "SYNC_HARMED_R_DOWN")]
    else:
        result = "NO_EFFECT"
        reasons = [str(rc.get("sync_no_effect") or "SYNC_NO_EFFECT")]
    return {
        "result": result,
        "reason_codes": reasons,
        "delta_r": round(delta, 6),
        "window_sec": int(window_sec),
        "evaluated_at_eval_ts_ms": int(evaluated_at_eval_ts_ms),
    }


def evaluate_downshift_state(
    *,
    outcomes: List[Mapping[str, Any]],
    now_ts_ms: int,
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    harm_threshold = int(policy.get("harm_streak_threshold", 2) or 2)
    unknown_threshold = int(policy.get("unknown_streak_threshold", 3) or 3)
    cooldown_sec = int(policy.get("cooldown_sec", 300) or 300)
    rc = (policy.get("reason_codes") or {}) if isinstance(policy.get("reason_codes"), Mapping) else {}
    ordered = sorted(
        [row for row in outcomes if isinstance(row, Mapping)],
        key=lambda row: int(row.get("evaluated_at_eval_ts_ms") or row.get("timestamp_ms") or 0),
    )
    harm_streak = 0
    unknown_streak = 0
    for row in reversed(ordered):
        result = str(row.get("result") or row.get("effect_result") or "").upper()
        if result == "HARMED":
            harm_streak += 1
            unknown_streak = 0
        elif result == "UNKNOWN":
            unknown_streak += 1
            harm_streak = 0
        else:
            break
    reasons: List[str] = []
    if harm_streak >= harm_threshold:
        reasons.append(str(rc.get("harm_streak") or "DOWNSHIFT_HARM_STREAK"))
    if unknown_streak >= unknown_threshold:
        reasons.append(str(rc.get("unknown_streak") or "DOWNSHIFT_UNKNOWN_STREAK"))
    applied = len(reasons) > 0
    cooldown_until = int(now_ts_ms + cooldown_sec * 1000) if applied else int(now_ts_ms)
    return {
        "applied": applied,
        "reason_codes": reasons,
        "cooldown_until_ts_ms": cooldown_until,
        "actions": [str(x) for x in (policy.get("actions") or [])] if applied else [],
    }


def is_sync_emit_suppressed(*, now_ts_ms: int, latest_downshift: Mapping[str, Any] | None) -> bool:
    if not isinstance(latest_downshift, Mapping):
        return False
    until = latest_downshift.get("cooldown_until_ts_ms")
    if not isinstance(until, (int, float)):
        return False
    return int(now_ts_ms) < int(until)
