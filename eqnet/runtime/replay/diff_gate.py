from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml


def load_diff_gate_policy(path: Path) -> Dict[str, Any]:
    default: Dict[str, Any] = {
        "schema_version": "diff_gate_policy_v0",
        "required_scope_keys": [
            "trace_path",
            "start_day_key",
            "end_day_key",
            "config_set_a",
            "config_set_b",
        ],
        "require_config_meta": True,
        "require_non_empty_top_regressions": True,
        "top_regression_allowed_families": ["harmed_rate", "unknown_rate", "blocked_rate"],
        "allowed_top_regression_metrics": [
            "sync_outcome_harmed_rate",
            "sync_outcome_unknown_rate",
            "sync_blocked_rate",
            "sync_micro_harmed_rate",
            "sync_micro_unknown_rate",
            "sync_downshift_applied_rate",
            "sync_emit_suppressed_time_ratio_avg",
        ],
        "disallow_improvement_metrics_in_top_regressions": [
            "sync_r_median_avg",
            "sync_r_p95_avg",
        ],
        "green_regression_guard": {
            "enabled": True,
            "blockers": {
                "sync_micro_harmed_rate": {
                    "max_delta": 0.0,
                    "reason_code": "GREEN_REGRESSION_SYNC_MICRO_HARMED_RATE",
                },
                "sync_downshift_applied_rate": {
                    "max_delta": 0.0,
                    "reason_code": "GREEN_REGRESSION_SYNC_DOWNSHIFT_APPLIED_RATE",
                },
                "sync_micro_unknown_rate": {
                    "max_delta": 0.0,
                    "reason_code": "GREEN_REGRESSION_SYNC_MICRO_UNKNOWN_RATE",
                },
            },
        },
        "behavior_change_guard": {
            "enabled": True,
            "blockers": {
                "behavior_change_harmed_rate_delta_avg": {
                    "max_delta": 0.03,
                    "reason_code": "BC_REGRESSION_HARMED_RATE_DELTA",
                },
                "behavior_change_reject_rate_delta_avg": {
                    "max_delta": 0.20,
                    "reason_code": "BC_REGRESSION_REJECT_RATE_DELTA",
                },
            },
        },
    }
    if not path.exists():
        return default
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return default
    if not isinstance(payload, dict):
        return default
    out = dict(default)
    out.update(payload)
    return out


def evaluate_diff_gate(diff_summary: Mapping[str, Any], policy: Mapping[str, Any]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    scope = diff_summary.get("comparison_scope")
    required_scope_keys = policy.get("required_scope_keys")
    missing_scope: List[str] = []
    if isinstance(required_scope_keys, list):
        scope_map = scope if isinstance(scope, Mapping) else {}
        for key in required_scope_keys:
            name = str(key or "").strip()
            if not name:
                continue
            value = scope_map.get(name)
            if value in (None, ""):
                missing_scope.append(name)
    checks.append(
        {
            "name": "comparison_scope_required",
            "ok": len(missing_scope) == 0,
            "detail": {"missing_scope_keys": missing_scope},
            "reason_code": "" if len(missing_scope) == 0 else "GATE_MISSING_SCOPE_KEY",
        }
    )

    require_config_meta = bool(policy.get("require_config_meta", True))
    missing_meta: List[str] = []
    if require_config_meta:
        for key in ("config_set_a_meta", "config_set_b_meta"):
            meta = diff_summary.get(key)
            files = meta.get("files") if isinstance(meta, Mapping) else None
            if not isinstance(files, Mapping) or len(files) == 0:
                missing_meta.append(key)
    checks.append(
        {
            "name": "config_meta_present",
            "ok": len(missing_meta) == 0,
            "detail": {"missing_config_meta": missing_meta},
            "reason_code": "" if len(missing_meta) == 0 else "GATE_MISSING_CONFIG_META",
        }
    )

    allowed_families = policy.get("top_regression_allowed_families")
    allowed = {str(x) for x in allowed_families} if isinstance(allowed_families, list) else {"harmed_rate", "unknown_rate", "blocked_rate"}
    allowed_metrics_raw = policy.get("allowed_top_regression_metrics")
    allowed_metrics = {str(x) for x in allowed_metrics_raw} if isinstance(allowed_metrics_raw, list) else set()
    disallow_raw = policy.get("disallow_improvement_metrics_in_top_regressions")
    disallow_metrics = {str(x) for x in disallow_raw} if isinstance(disallow_raw, list) else set()
    top_regressions = diff_summary.get("top_regressions")
    first_metric = ""
    first_family = ""
    empty_required = bool(policy.get("require_non_empty_top_regressions", True))
    if isinstance(top_regressions, list) and len(top_regressions) > 0 and isinstance(top_regressions[0], Mapping):
        first_metric = str(top_regressions[0].get("metric") or "")
        first_family = _metric_family(first_metric)
    top_regression_ok = True
    top_regression_reason = ""
    if empty_required and not first_metric:
        top_regression_ok = False
        top_regression_reason = "GATE_TOP_REGRESSION_NOT_RISK_METRIC"
    elif first_metric:
        if first_metric in disallow_metrics:
            top_regression_ok = False
            top_regression_reason = "GATE_TOP_REGRESSION_IS_IMPROVEMENT_METRIC"
        else:
            top_regression_ok = (first_metric in allowed_metrics) or (first_family in allowed)
            if not top_regression_ok:
                top_regression_reason = "GATE_TOP_REGRESSION_NOT_RISK_METRIC"
    checks.append(
        {
            "name": "top_regression_priority",
            "ok": top_regression_ok,
            "detail": {
                "first_metric": first_metric,
                "first_metric_family": first_family,
                "allowed_families": sorted(allowed),
                "allowed_metrics": sorted(allowed_metrics),
                "disallow_metrics": sorted(disallow_metrics),
            },
            "reason_code": "" if top_regression_ok else top_regression_reason,
        }
    )

    green_guard = policy.get("green_regression_guard")
    green_guard_enabled = bool(green_guard.get("enabled")) if isinstance(green_guard, Mapping) else False
    violations: List[Dict[str, Any]] = []
    green_reason_codes: List[str] = []
    if green_guard_enabled:
        blockers = green_guard.get("blockers") if isinstance(green_guard, Mapping) else None
        aggregate_delta = diff_summary.get("aggregate_delta")
        blocker_map = blockers if isinstance(blockers, Mapping) else {}
        delta_map = aggregate_delta if isinstance(aggregate_delta, Mapping) else {}
        for metric_key, spec in blocker_map.items():
            metric = str(metric_key or "").strip()
            if not metric:
                continue
            if metric not in delta_map:
                continue
            value = delta_map.get(metric)
            try:
                delta = float(value)
            except (TypeError, ValueError):
                continue
            max_delta = 0.0
            reason_code = "GREEN_REGRESSION_GUARD_BLOCKED"
            if isinstance(spec, Mapping):
                try:
                    max_delta = float(spec.get("max_delta", 0.0))
                except (TypeError, ValueError):
                    max_delta = 0.0
                candidate = str(spec.get("reason_code") or "").strip()
                if candidate:
                    reason_code = candidate
            if delta > max_delta:
                violations.append({"metric": metric, "delta": delta, "max_delta": max_delta, "reason_code": reason_code})
                green_reason_codes.append(reason_code)
    checks.append(
        {
            "name": "green_regression_guard",
            "ok": len(violations) == 0,
            "detail": {"enabled": green_guard_enabled, "violations": violations},
            "reason_code": "" if len(violations) == 0 else sorted(set(green_reason_codes))[0],
            "extra_reason_codes": sorted(set(green_reason_codes)),
        }
    )

    behavior_guard = policy.get("behavior_change_guard")
    behavior_guard_enabled = bool(behavior_guard.get("enabled")) if isinstance(behavior_guard, Mapping) else False
    behavior_violations: List[Dict[str, Any]] = []
    behavior_reason_codes: List[str] = []
    if behavior_guard_enabled:
        blockers = behavior_guard.get("blockers") if isinstance(behavior_guard, Mapping) else None
        aggregate_delta = diff_summary.get("aggregate_delta")
        blocker_map = blockers if isinstance(blockers, Mapping) else {}
        delta_map = aggregate_delta if isinstance(aggregate_delta, Mapping) else {}
        for metric_key, spec in blocker_map.items():
            metric = str(metric_key or "").strip()
            if not metric:
                continue
            if metric not in delta_map:
                continue
            value = delta_map.get(metric)
            try:
                delta = float(value)
            except (TypeError, ValueError):
                continue
            max_delta = 0.0
            reason_code = "BC_REGRESSION_GUARD_BLOCKED"
            if isinstance(spec, Mapping):
                try:
                    max_delta = float(spec.get("max_delta", 0.0))
                except (TypeError, ValueError):
                    max_delta = 0.0
                candidate = str(spec.get("reason_code") or "").strip()
                if candidate:
                    reason_code = candidate
            if delta > max_delta:
                behavior_violations.append({"metric": metric, "delta": delta, "max_delta": max_delta, "reason_code": reason_code})
                behavior_reason_codes.append(reason_code)
    checks.append(
        {
            "name": "behavior_change_guard",
            "ok": len(behavior_violations) == 0,
            "detail": {"enabled": behavior_guard_enabled, "violations": behavior_violations},
            "reason_code": "" if len(behavior_violations) == 0 else sorted(set(behavior_reason_codes))[0],
            "extra_reason_codes": sorted(set(behavior_reason_codes)),
        }
    )

    errors = [str(c.get("name") or "unknown_check") for c in checks if not bool(c.get("ok"))]
    reason_codes: List[str] = []
    for check in checks:
        if bool(check.get("ok")):
            continue
        base_reason = str(check.get("reason_code") or "").strip()
        if base_reason:
            reason_codes.append(base_reason)
        extra = check.get("extra_reason_codes")
        if isinstance(extra, list):
            for value in extra:
                code = str(value or "").strip()
                if code:
                    reason_codes.append(code)
    comparison_scope = diff_summary.get("comparison_scope")
    eval_ts_ms = None
    if isinstance(comparison_scope, Mapping):
        value = comparison_scope.get("eval_ts_ms")
        if isinstance(value, (int, float)):
            eval_ts_ms = int(value)
    unique_reason_codes = sorted(set(reason_codes))
    severity = "BLOCKER" if unique_reason_codes else "OK"
    exit_code = 0 if len(errors) == 0 else 2
    return {
        "schema_version": "gate_result_v0",
        "ok": len(errors) == 0,
        "exit_code": exit_code,
        "severity": severity,
        "checks": checks,
        "errors": errors,
        "reason_codes": unique_reason_codes,
        "gate_policy_fingerprint": _fingerprint_payload(policy),
        "diff_summary_fingerprint": _fingerprint_payload(diff_summary),
        "evaluated_at_eval_ts_ms": eval_ts_ms,
        "comparison_scope": dict(comparison_scope) if isinstance(comparison_scope, Mapping) else {},
    }


def render_gate_markdown(result: Mapping[str, Any]) -> str:
    ok = bool(result.get("ok"))
    lines = ["# Replay Diff Gate", ""]
    lines.append(f"- overall_ok: {str(ok).lower()}")
    schema_version = str(result.get("schema_version") or "")
    if schema_version:
        lines.append(f"- schema_version: {schema_version}")
    lines.append(f"- exit_code: {int(result.get('exit_code') or (0 if ok else 2))}")
    lines.append(f"- severity: {str(result.get('severity') or ('OK' if ok else 'BLOCKER'))}")
    policy_fp = str(result.get("gate_policy_fingerprint") or "")
    summary_fp = str(result.get("diff_summary_fingerprint") or "")
    eval_ts = result.get("evaluated_at_eval_ts_ms")
    if policy_fp:
        lines.append(f"- gate_policy_fingerprint: {policy_fp}")
    if summary_fp:
        lines.append(f"- diff_summary_fingerprint: {summary_fp}")
    if isinstance(eval_ts, int):
        lines.append(f"- evaluated_at_eval_ts_ms: {eval_ts}")
    for check in result.get("checks") or []:
        if not isinstance(check, Mapping):
            continue
        name = str(check.get("name") or "unknown_check")
        status = "ok" if bool(check.get("ok")) else "ng"
        lines.append(f"- {name}: {status}")
    reason_codes = result.get("reason_codes")
    if isinstance(reason_codes, list) and len(reason_codes) > 0:
        lines.append("- reason_codes:")
        for code in reason_codes:
            lines.append(f"  - {str(code)}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _metric_family(metric_key: str) -> str:
    key = str(metric_key).lower()
    if "harmed" in key:
        return "harmed_rate"
    if "unknown" in key:
        return "unknown_rate"
    if "helped" in key:
        return "helped_rate"
    return "other"


def _normalize_for_hash(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _normalize_for_hash(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_normalize_for_hash(v) for v in value]
    if isinstance(value, tuple):
        return [_normalize_for_hash(v) for v in value]
    return value


def _fingerprint_payload(payload: Mapping[str, Any] | Dict[str, Any]) -> str:
    canonical = json.dumps(_normalize_for_hash(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
