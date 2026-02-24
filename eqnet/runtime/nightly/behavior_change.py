from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import yaml


def load_behavior_change_policy(path: Path) -> Dict[str, Any]:
    default: Dict[str, Any] = {
        "schema_version": "behavior_change_v1",
        "enabled": True,
        "window": {
            "baseline_days": 7,
            "compare_days": 1,
        },
        "diff_gate": {
            "thresholds": {
                "harmed_rate_delta": {"warn": 0.01, "fail": 0.03},
                "reject_rate_delta": {"warn": 0.10, "fail": 0.20},
            },
            "min_support": {"per_signature_events": 1},
        },
        "decision": {
            "reject_values": ["REJECT"],
            "accept_values": ["ACCEPT_SHADOW", "ACCEPT_CANARY", "ACCEPT_ROLLOUT", "ROLLBACK", "LINK_EVAL_REPORT"],
        },
        "signature": {
            "fields": ["world_type", "risk_bucket", "uncertainty_bucket", "response_route"],
            "max_keys": 128,
            "hash_key": False,
            "field_sanitizers": {},
        },
        "tolerance": {
            "enabled": True,
            "apply_to": "warn_only",
            "active_preset": "default",
            "preset_source": "manual",
            "margin_cap": 0.02,
            "recovery_alpha": 0.6,
            "harm_beta": 0.8,
            "unknown_gamma": 0.8,
            "reject_beta": 0.4,
            "epsilon_to_fail": 0.001,
            "initial_trust_budget": 0.5,
            "mix_weight_sig": 0.7,
            "min_support_per_sig": 1,
            "trust_decay": 0.02,
            "max_daily_delta": 0.05,
            "presets": {
                "default": {"margin_cap": 0.02, "recovery_alpha": 0.6, "mix_weight_sig": 0.7},
                "love": {"margin_cap": 0.03, "recovery_alpha": 0.75, "mix_weight_sig": 0.85},
                "recovery": {"margin_cap": 0.015, "recovery_alpha": 0.5, "mix_weight_sig": 0.6},
            },
        },
    }
    if not path.exists():
        return default
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return default
    if not isinstance(payload, Mapping):
        return default
    merged = dict(default)
    for key, value in payload.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            current = dict(merged[key])  # type: ignore[index]
            current.update(dict(value))
            merged[key] = current
        else:
            merged[key] = value
    return merged


def compute_behavior_change_snapshot(
    *,
    current_payload: Mapping[str, Any],
    prev_payload: Mapping[str, Any] | None,
    telemetry_dir: Path,
    day: date,
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    if not bool(policy.get("enabled", True)):
        return {
            "schema_version": "behavior_change_snapshot_v1",
            "enabled": False,
            "proposal_priority_shift": 0.0,
            "reject_rate_delta": 0.0,
            "harmed_rate_delta": 0.0,
            "gate": {"status": "PASS", "reason_codes": []},
            "support": {"baseline_events": 0, "compare_events": 0},
        }

    baseline_days = _safe_int(((policy.get("window") or {}).get("baseline_days")), 7)
    compare_days = _safe_int(((policy.get("window") or {}).get("compare_days")), 1)
    baseline_rows = _load_decision_rows(telemetry_dir, _days_before(day, baseline_days, skip_today=True))
    compare_rows = _load_decision_rows(telemetry_dir, _days_before(day, compare_days, skip_today=False))

    decision_policy = policy.get("decision") if isinstance(policy.get("decision"), Mapping) else {}
    reject_values = _as_upper_set((decision_policy or {}).get("reject_values"), {"REJECT"})
    accept_values = _as_upper_set(
        (decision_policy or {}).get("accept_values"),
        {"ACCEPT_SHADOW", "ACCEPT_CANARY", "ACCEPT_ROLLOUT", "ROLLBACK", "LINK_EVAL_REPORT"},
    )

    baseline_reject_rate = _reject_rate(baseline_rows, reject_values=reject_values, accept_values=accept_values)
    compare_reject_rate = _reject_rate(compare_rows, reject_values=reject_values, accept_values=accept_values)
    reject_rate_delta = round(compare_reject_rate - baseline_reject_rate, 6)
    signature_cfg = policy.get("signature") if isinstance(policy.get("signature"), Mapping) else {}
    signature_fields = signature_cfg.get("fields") if isinstance(signature_cfg.get("fields"), list) else ["world_type", "risk_bucket", "uncertainty_bucket", "response_route"]
    max_keys = max(1, _safe_int(signature_cfg.get("max_keys"), 128))
    hash_key = bool(signature_cfg.get("hash_key", False))
    sanitizers = signature_cfg.get("field_sanitizers") if isinstance(signature_cfg.get("field_sanitizers"), Mapping) else {}
    sig_info = _signature_from_mapping(current_payload, signature_fields, sanitizers=sanitizers, hash_key=hash_key)
    signature_key = str(sig_info.get("key") or "sig:global")
    baseline_sig_rows = _filter_rows_by_signature(
        baseline_rows,
        signature_fields,
        signature_key,
        sanitizers=sanitizers,
        hash_key=hash_key,
    )
    compare_sig_rows = _filter_rows_by_signature(
        compare_rows,
        signature_fields,
        signature_key,
        sanitizers=sanitizers,
        hash_key=hash_key,
    )
    active_sig_keys = _collect_lru_signature_keys(
        compare_rows,
        fields=signature_fields,
        max_keys=max_keys,
        sanitizers=sanitizers,
        hash_key=hash_key,
    )
    baseline_reject_rate_sig = _reject_rate(baseline_sig_rows, reject_values=reject_values, accept_values=accept_values)
    compare_reject_rate_sig = _reject_rate(compare_sig_rows, reject_values=reject_values, accept_values=accept_values)
    reject_rate_delta_sig = round(compare_reject_rate_sig - baseline_reject_rate_sig, 6)

    current_priority = _priority_median(current_payload)
    prev_priority = _priority_median(prev_payload or {})
    proposal_priority_shift = round(current_priority - prev_priority, 6)

    current_harmed_rate = _harmed_rate(current_payload)
    prev_harmed_rate = _harmed_rate(prev_payload or {})
    harmed_rate_delta = round(current_harmed_rate - prev_harmed_rate, 6)

    thresholds = ((policy.get("diff_gate") or {}).get("thresholds") or {}) if isinstance(policy.get("diff_gate"), Mapping) else {}
    min_support = _safe_int((((policy.get("diff_gate") or {}).get("min_support") or {}).get("per_signature_events")), 1)
    compare_support = len(compare_rows)
    baseline_support = len(baseline_rows)

    reason_codes: List[str] = []
    status = "PASS"
    if compare_support < min_support:
        status = "WARN"
        reason_codes.append("BC_UNDERSAMPLED_COMPARE")
    if baseline_support < min_support:
        status = "WARN" if status == "PASS" else status
        reason_codes.append("BC_UNDERSAMPLED_BASELINE")

    harmed_fail = _threshold_value(thresholds, "harmed_rate_delta", "fail", 0.03)
    harmed_warn = _threshold_value(thresholds, "harmed_rate_delta", "warn", 0.01)
    reject_fail = _threshold_value(thresholds, "reject_rate_delta", "fail", 0.20)
    reject_warn = _threshold_value(thresholds, "reject_rate_delta", "warn", 0.10)
    tolerance = policy.get("tolerance") if isinstance(policy.get("tolerance"), Mapping) else {}
    tolerance_enabled = bool((tolerance or {}).get("enabled", True))
    tolerance_apply_to = str((tolerance or {}).get("apply_to") or "warn_only")
    preset_source = str((tolerance or {}).get("preset_source") or "manual")
    tolerance_presets = (tolerance or {}).get("presets") if isinstance((tolerance or {}).get("presets"), Mapping) else {}
    requested_preset = str((tolerance or {}).get("active_preset") or "default")
    if isinstance(tolerance_presets, Mapping) and requested_preset in tolerance_presets and isinstance(tolerance_presets.get(requested_preset), Mapping):
        active_preset = requested_preset
    elif isinstance(tolerance_presets, Mapping) and isinstance(tolerance_presets.get("default"), Mapping):
        active_preset = "default"
    else:
        active_preset = "base"
    selected_preset = tolerance_presets.get(active_preset) if isinstance(tolerance_presets, Mapping) and isinstance(tolerance_presets.get(active_preset), Mapping) else {}
    trust_budget_global = _safe_float((tolerance or {}).get("initial_trust_budget"), 0.5)
    trust_budget_sig = trust_budget_global
    effective_trust_budget = trust_budget_global
    trust_source = "global"
    margin = 0.0
    effective_harmed_warn = harmed_warn
    effective_reject_warn = reject_warn
    if tolerance_enabled:
        base_margin_cap = max(0.0, _safe_float((tolerance or {}).get("margin_cap"), 0.02))
        base_recovery_alpha = max(0.0, _safe_float((tolerance or {}).get("recovery_alpha"), 0.6))
        base_mix_weight_sig = _clip(_safe_float((tolerance or {}).get("mix_weight_sig"), 0.7), 0.0, 1.0)
        margin_cap = max(0.0, _safe_float((selected_preset or {}).get("margin_cap"), base_margin_cap))
        recovery_alpha = max(0.0, _safe_float((selected_preset or {}).get("recovery_alpha"), base_recovery_alpha))
        harm_beta = max(0.0, _safe_float((tolerance or {}).get("harm_beta"), 0.8))
        unknown_gamma = max(0.0, _safe_float((tolerance or {}).get("unknown_gamma"), harm_beta))
        reject_beta = max(0.0, _safe_float((tolerance or {}).get("reject_beta"), 0.4))
        mix_weight_sig = _clip(_safe_float((selected_preset or {}).get("mix_weight_sig"), base_mix_weight_sig), 0.0, 1.0)
        min_support_per_sig = max(0, _safe_int((tolerance or {}).get("min_support_per_sig"), 1))
        trust_decay = _clip(_safe_float((tolerance or {}).get("trust_decay"), 0.02), 0.0, 1.0)
        max_daily_delta = max(0.0, _safe_float((tolerance or {}).get("max_daily_delta"), 0.05))
        epsilon_to_fail = max(0.0, _safe_float((tolerance or {}).get("epsilon_to_fail"), 0.001))
        helped_rate = _safe_float(current_payload.get("sync_micro_helped_rate"), 0.0)
        unknown_rate = _safe_float(current_payload.get("sync_micro_unknown_rate"), 0.0)
        delta_raw_global = (recovery_alpha * helped_rate) - (harm_beta * current_harmed_rate) - (unknown_gamma * unknown_rate)
        delta_global = _clip(delta_raw_global, -max_daily_delta, max_daily_delta)
        trust_budget_global = _clip(
            trust_budget_global + delta_global,
            0.0,
            1.0,
        )
        trust_budget_global = _clip(
            trust_budget_global + (trust_decay * (0.5 - trust_budget_global)),
            0.0,
            1.0,
        )
        helped_sig_rate = max(0.0, 1.0 - compare_reject_rate_sig)
        delta_raw_sig = (
            (recovery_alpha * helped_sig_rate)
            - (reject_beta * compare_reject_rate_sig)
            - (harm_beta * current_harmed_rate)
            - (unknown_gamma * unknown_rate)
        )
        delta_sig = _clip(delta_raw_sig, -max_daily_delta, max_daily_delta)
        trust_budget_sig = _clip(
            _safe_float((tolerance or {}).get("initial_trust_budget"), 0.5)
            + delta_sig,
            0.0,
            1.0,
        )
        trust_budget_sig = _clip(
            trust_budget_sig + (trust_decay * (0.5 - trust_budget_sig)),
            0.0,
            1.0,
        )
        sig_supported = len(compare_sig_rows) >= min_support_per_sig
        sig_active = signature_key in active_sig_keys
        if sig_supported and sig_active:
            effective_trust_budget = _clip((mix_weight_sig * trust_budget_sig) + ((1.0 - mix_weight_sig) * trust_budget_global), 0.0, 1.0)
            trust_source = "mixed"
        else:
            effective_trust_budget = trust_budget_global
            trust_source = "global_fallback" if sig_supported else "global_fallback_min_support"
        # trust ↑ => margin+, so WARN threshold becomes looser.
        margin = _clip(margin_cap * ((2.0 * effective_trust_budget) - 1.0), -margin_cap, margin_cap)
        if tolerance_apply_to == "warn_only":
            effective_harmed_warn = _clip(harmed_warn + margin, 0.0, max(0.0, harmed_fail - epsilon_to_fail))
            effective_reject_warn = _clip(reject_warn + margin, 0.0, max(0.0, reject_fail - epsilon_to_fail))

    if harmed_rate_delta > harmed_fail:
        status = "FAIL"
        reason_codes.append("BC_HARMED_RATE_DELTA_FAIL")
    elif harmed_rate_delta > effective_harmed_warn and status != "FAIL":
        status = "WARN"
        reason_codes.append("BC_HARMED_RATE_DELTA_WARN")

    if reject_rate_delta > reject_fail:
        status = "FAIL"
        reason_codes.append("BC_REJECT_RATE_DELTA_FAIL")
    elif reject_rate_delta > effective_reject_warn and status != "FAIL":
        status = "WARN"
        reason_codes.append("BC_REJECT_RATE_DELTA_WARN")

    return {
        "schema_version": "behavior_change_snapshot_v1",
        "enabled": True,
        "proposal_priority_shift": proposal_priority_shift,
        "reject_rate_delta": reject_rate_delta,
        "harmed_rate_delta": harmed_rate_delta,
        "gate": {"status": status, "reason_codes": sorted(set(reason_codes))},
        "support": {"baseline_events": baseline_support, "compare_events": compare_support},
        "baseline": {"reject_rate": round(baseline_reject_rate, 6), "harmed_rate": round(prev_harmed_rate, 6)},
        "compare": {"reject_rate": round(compare_reject_rate, 6), "harmed_rate": round(current_harmed_rate, 6)},
        "signature": {
            "key": signature_key,
            "values": dict(sig_info.get("values") or {}),
            "fields": [str(v) for v in signature_fields],
            "max_keys": int(max_keys),
            "active_key_count": int(len(active_sig_keys)),
        },
        "signature_support": {
            "baseline_events": len(baseline_sig_rows),
            "compare_events": len(compare_sig_rows),
        },
        "signature_delta": {
            "reject_rate_delta": reject_rate_delta_sig,
        },
        "tolerance": {
            "enabled": bool(tolerance_enabled),
            "apply_to": tolerance_apply_to,
            "active_preset": active_preset,
            "preset_source": preset_source,
            "trust_budget_global": round(trust_budget_global, 6),
            "trust_budget_sig": round(trust_budget_sig, 6),
            "effective_trust_budget": round(effective_trust_budget, 6),
            "trust_source": trust_source,
            "mix_weight_sig_effective": round(
                mix_weight_sig if trust_source == "mixed" else 0.0,
                6,
            ),
            "margin": round(margin, 6),
            "effective_warn": {
                "harmed_rate_delta": round(effective_harmed_warn, 6),
                "reject_rate_delta": round(effective_reject_warn, 6),
            },
            "base_warn": {
                "harmed_rate_delta": round(harmed_warn, 6),
                "reject_rate_delta": round(reject_warn, 6),
            },
            "base_fail": {
                "harmed_rate_delta": round(harmed_fail, 6),
                "reject_rate_delta": round(reject_fail, 6),
            },
        },
    }


def _days_before(target: date, count: int, *, skip_today: bool) -> List[date]:
    days: List[date] = []
    if count <= 0:
        return days
    start = 1 if skip_today else 0
    for offset in range(start, start + count):
        days.append(target - timedelta(days=offset))
    return sorted(days)


def _load_decision_rows(telemetry_dir: Path, days: Iterable[date]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for day in days:
        path = telemetry_dir / f"change_decisions-{day.strftime('%Y%m%d')}.jsonl"
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, Mapping):
                    continue
                if str(obj.get("schema_version") or "") != "change_decision.v0":
                    continue
                rows.append(dict(obj))
        except Exception:
            continue
    return rows


def _as_upper_set(values: Any, default: set[str]) -> set[str]:
    if not isinstance(values, list):
        return set(default)
    out = {str(value or "").strip().upper() for value in values}
    out.discard("")
    return out or set(default)


def _reject_rate(rows: List[Mapping[str, Any]], *, reject_values: set[str], accept_values: set[str]) -> float:
    if not rows:
        return 0.0
    total = 0
    reject = 0
    allowed = set(reject_values) | set(accept_values)
    for row in rows:
        decision = str(row.get("decision") or "").strip().upper()
        if decision not in allowed:
            continue
        total += 1
        if decision in reject_values:
            reject += 1
    if total <= 0:
        return 0.0
    return float(reject) / float(total)


def _priority_median(payload: Mapping[str, Any]) -> float:
    stats = payload.get("priority_score_stats")
    if not isinstance(stats, Mapping):
        return 0.0
    value = stats.get("median")
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _harmed_rate(payload: Mapping[str, Any]) -> float:
    for key in ("sync_micro_harmed_rate", "realtime_harmed_rate"):
        value = payload.get(key)
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _threshold_value(thresholds: Mapping[str, Any], name: str, level: str, default: float) -> float:
    node = thresholds.get(name) if isinstance(thresholds, Mapping) else None
    if not isinstance(node, Mapping):
        return float(default)
    value = node.get(level)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _signature_from_mapping(
    payload: Mapping[str, Any],
    fields: List[Any],
    *,
    sanitizers: Mapping[str, Any] | None = None,
    hash_key: bool = False,
) -> Dict[str, Any]:
    values: Dict[str, str] = {}
    for raw in fields:
        key = str(raw or "").strip()
        if not key:
            continue
        value = payload.get(key, "unknown")
        values[key] = _sanitize_signature_value(key, value, sanitizers=sanitizers)
    parts = [f"{key}={values[key]}" for key in sorted(values.keys())]
    plain_key = "sig:" + "|".join(parts) if parts else "sig:global"
    sig_key = plain_key
    if hash_key:
        digest = hashlib.sha256(plain_key.encode("utf-8")).hexdigest()[:16]
        sig_key = f"sig#{digest}"
    return {"key": sig_key, "values": values}


def _filter_rows_by_signature(
    rows: List[Mapping[str, Any]],
    fields: List[Any],
    signature_key: str,
    *,
    sanitizers: Mapping[str, Any] | None = None,
    hash_key: bool = False,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        info = _signature_from_mapping(row, fields, sanitizers=sanitizers, hash_key=hash_key)
        if str(info.get("key") or "") == signature_key:
            out.append(dict(row))
    return out


def _collect_lru_signature_keys(
    rows: List[Mapping[str, Any]],
    *,
    fields: List[Any],
    max_keys: int,
    sanitizers: Mapping[str, Any] | None = None,
    hash_key: bool = False,
) -> List[str]:
    touched: List[str] = []
    ordered = sorted(rows, key=lambda row: int(_safe_int(row.get("timestamp_ms"), 0)))
    for row in ordered:
        key = str(
            _signature_from_mapping(
                row,
                fields,
                sanitizers=sanitizers,
                hash_key=hash_key,
            ).get("key")
            or "sig:global"
        )
        if key in touched:
            touched.remove(key)
        touched.append(key)
        if len(touched) > max_keys:
            touched.pop(0)
    return touched


def _sanitize_signature_value(
    key: str,
    value: Any,
    *,
    sanitizers: Mapping[str, Any] | None = None,
) -> str:
    specs = sanitizers if isinstance(sanitizers, Mapping) else {}
    spec = specs.get(key) if isinstance(specs, Mapping) else None
    empty_value = "unknown"
    kind = "identity"
    if isinstance(spec, str):
        kind = spec
    elif isinstance(spec, Mapping):
        kind = str(spec.get("kind") or "identity").strip().lower()
        empty_value = str(spec.get("empty_value") or "unknown")
    raw = value
    if raw is None or str(raw).strip() == "":
        return empty_value
    if kind == "lower":
        return str(raw).strip().lower()
    if kind == "bucket":
        if not isinstance(spec, Mapping):
            return str(raw)
        width = max(1e-9, _safe_float(spec.get("width"), 0.1))
        low = _safe_float(spec.get("min"), 0.0)
        high = _safe_float(spec.get("max"), 1.0)
        numeric = _clip(_safe_float(raw, low), low, high)
        idx = int((numeric - low) // width)
        bucket = low + (idx * width)
        return f"{bucket:.6f}"
    if kind == "round":
        decimals = max(0, _safe_int(spec.get("decimals"), 2) if isinstance(spec, Mapping) else 2)
        numeric = _safe_float(raw, 0.0)
        return f"{round(numeric, decimals):.{decimals}f}"
    return str(raw).strip()


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clip(value: float, min_value: float, max_value: float) -> float:
    if value < min_value:
        return float(min_value)
    if value > max_value:
        return float(max_value)
    return float(value)
