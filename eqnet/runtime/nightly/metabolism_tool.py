from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping


DEFAULT_METABOLISM_POLICY: Dict[str, Any] = {
    "policy_version": "memory-ops-v1",
    "entropy_model_id": "defrag-observe-v1",
    "enabled_metrics": [
        "memory_item_count",
        "link_count",
        "bytes_estimate",
        "summary_count",
    ],
    "delta_weights": {
        "memory_item_count": 1.0,
        "link_count": 1.0,
        "bytes_estimate": 1.0,
        "summary_count": 1.0,
    },
    "entropy_cost_class_thresholds": {
        "mid": 0.05,
        "high": 0.20,
    },
    "default_phase": "stabilization",
    "phase_profiles": {
        "exploration": {
            "phase_weight_profile": "phase.exploration.v1",
            "delta_weights": {
                "memory_item_count": 1.3,
                "link_count": 1.1,
                "bytes_estimate": 0.8,
                "summary_count": 0.7,
            },
            "entropy_cost_class_thresholds": {"mid": 0.06, "high": 0.22},
        },
        "stabilization": {
            "phase_weight_profile": "phase.stabilization.v1",
            "delta_weights": {
                "memory_item_count": 1.0,
                "link_count": 1.0,
                "bytes_estimate": 1.0,
                "summary_count": 1.0,
            },
            "entropy_cost_class_thresholds": {"mid": 0.05, "high": 0.20},
        },
        "recovery": {
            "phase_weight_profile": "phase.recovery.v1",
            "delta_weights": {
                "memory_item_count": 0.8,
                "link_count": 1.4,
                "bytes_estimate": 1.1,
                "summary_count": 1.2,
            },
            "entropy_cost_class_thresholds": {"mid": 0.04, "high": 0.16},
        },
    },
    "energy_budget_limit": 0.10,
    "output_control_profiles": {
        "normal": "normal_v1",
        "throttled": "cautious_budget_v1",
    },
    "throttle_reason_codes": {
        "budget_exceeded": "BUDGET_EXCEEDED",
    },
    "resource_budget": {
        "entropy": {
            "capacity": 1.0,
            "spend_per_record": 0.0,
            "recover_per_night": 0.02,
        },
        "attention": {
            "capacity": 1.0,
            "spend_per_record": 0.004,
            "recover_per_night": 0.05,
        },
        "affect": {
            "capacity": 1.0,
            "spend_per_record": 0.003,
            "recover_per_night": 0.04,
        },
    },
}


def load_metabolism_policy(config: Any) -> Dict[str, Any]:
    raw = getattr(config, "memory_thermo_policy", None)
    if not isinstance(raw, Mapping):
        return _deep_merge(DEFAULT_METABOLISM_POLICY, {})
    return _deep_merge(DEFAULT_METABOLISM_POLICY, raw)


def run_metabolism_cycle(
    *,
    qualia_records: list[dict[str, Any]],
    policy: Mapping[str, Any],
    previous_state: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    phase_ctx = _resolve_phase_context(policy, previous_state)
    before = _measure_defrag_metrics(qualia_records, policy)
    # Stage-1 observe: no structural mutation, only metering.
    after = dict(before)
    delta = _metrics_delta(before, after)
    memory_entropy_delta = _entropy_delta_from_metrics(
        delta,
        phase_ctx.get("delta_weights") or {},
    )
    entropy_cost_class = _entropy_cost_class(
        memory_entropy_delta,
        phase_ctx.get("entropy_cost_class_thresholds") or {},
    )
    resource_budget = _resource_budget_state(
        record_count=len(qualia_records),
        policy=policy,
        previous_state=previous_state,
    )
    entropy_spent = float((resource_budget.get("entropy") or {}).get("spent") or 0.0)
    energy_budget_used = float(memory_entropy_delta + entropy_spent)
    energy_budget_limit = float(policy.get("energy_budget_limit", 0.10))
    budget_throttle_applied = energy_budget_used >= energy_budget_limit
    entropy_budget_ok = energy_budget_used <= energy_budget_limit
    profile_cfg = (
        policy.get("output_control_profiles")
        if isinstance(policy.get("output_control_profiles"), Mapping)
        else {}
    )
    reason_cfg = (
        policy.get("throttle_reason_codes")
        if isinstance(policy.get("throttle_reason_codes"), Mapping)
        else {}
    )
    output_control_profile = str(
        profile_cfg.get("throttled" if budget_throttle_applied else "normal")
        or ("cautious_budget_v1" if budget_throttle_applied else "normal_v1")
    )
    throttle_reason_code = (
        str(reason_cfg.get("budget_exceeded") or "BUDGET_EXCEEDED")
        if budget_throttle_applied
        else ""
    )
    projection_fp = _value_projection_fingerprint(
        policy_version=str(policy.get("policy_version") or "memory-ops-v1"),
        entropy_model_id=str(policy.get("entropy_model_id") or "defrag-observe-v1"),
        memory_phase=str(phase_ctx.get("memory_phase") or "stabilization"),
        phase_weight_profile=str(phase_ctx.get("phase_weight_profile") or "default"),
        delta_weights=phase_ctx.get("delta_weights") or {},
        thresholds=phase_ctx.get("entropy_cost_class_thresholds") or {},
    )
    attention = _mapping(resource_budget.get("attention"))
    affect = _mapping(resource_budget.get("affect"))
    invariants = _metabolism_invariants(resource_budget)
    metabolism_invariants_ok = bool(invariants.get("ok", False))
    metabolism_conservation_error = float(invariants.get("max_conservation_error", 0.0))
    return {
        "defrag_status": "observed",
        "defrag_mode": "observe",
        "defrag_metrics_before": before,
        "defrag_metrics_after": after,
        "defrag_metrics_delta": delta,
        "memory_entropy_delta": memory_entropy_delta,
        "entropy_cost_class": entropy_cost_class,
        "irreversible_op": False,
        "entropy_budget_ok": entropy_budget_ok,
        "memory_phase": str(phase_ctx.get("memory_phase") or "stabilization"),
        "phase_weight_profile": str(phase_ctx.get("phase_weight_profile") or "default"),
        "value_projection_fingerprint": projection_fp,
        "phase_override_applied": bool(phase_ctx.get("phase_override_applied", False)),
        "energy_budget_used": energy_budget_used,
        "energy_budget_limit": energy_budget_limit,
        "budget_throttle_applied": budget_throttle_applied,
        "output_control_profile": output_control_profile,
        "throttle_reason_code": throttle_reason_code,
        "policy_version": str(policy.get("policy_version") or "memory-ops-v1"),
        "entropy_model_id": str(policy.get("entropy_model_id") or "defrag-observe-v1"),
        "metabolism_status": "applied",
        "metabolism_tool_version": "metabolism_tool_v1",
        "attention_budget_level": float(attention.get("level", 0.0)),
        "attention_budget_used": float(attention.get("spent", 0.0)),
        "attention_budget_recovered": float(attention.get("recovered", 0.0)),
        "affect_budget_level": float(affect.get("level", 0.0)),
        "affect_budget_used": float(affect.get("spent", 0.0)),
        "affect_budget_recovered": float(affect.get("recovered", 0.0)),
        "resource_budget": resource_budget,
        "metabolism_invariants_ok": metabolism_invariants_ok,
        "metabolism_conservation_error": metabolism_conservation_error,
        "metabolism_invariants": invariants,
    }


def _measure_defrag_metrics(
    qualia_records: list[dict[str, Any]],
    policy: Mapping[str, Any],
) -> Dict[str, float]:
    enabled = policy.get("enabled_metrics") or []
    metrics: Dict[str, float] = {}
    for metric_name in enabled:
        key = str(metric_name)
        if key == "memory_item_count":
            metrics[key] = float(len(qualia_records))
        elif key == "link_count":
            metrics[key] = float(
                sum(
                    1
                    for row in qualia_records
                    if isinstance(row, Mapping)
                    and any(
                        candidate in row
                        for candidate in ("parent_id", "link_id", "edge_id", "ref_id")
                    )
                )
            )
        elif key == "bytes_estimate":
            metrics[key] = float(
                sum(len(json.dumps(row, ensure_ascii=False, default=str)) for row in qualia_records)
            )
        elif key == "summary_count":
            metrics[key] = float(
                sum(
                    1
                    for row in qualia_records
                    if isinstance(row, Mapping)
                    and str(row.get("type") or row.get("kind") or "").lower() == "summary"
                )
            )
    return metrics


def _metrics_delta(before: Mapping[str, float], after: Mapping[str, float]) -> Dict[str, float]:
    keys = set(before.keys()) | set(after.keys())
    return {k: float(after.get(k, 0.0) - before.get(k, 0.0)) for k in sorted(keys)}


def _entropy_delta_from_metrics(delta: Mapping[str, float], weights: Mapping[str, Any]) -> float:
    total = 0.0
    scale = 0.0
    for key, value in delta.items():
        weight = float(weights.get(key, 1.0))
        total += abs(float(value)) * weight
        scale += abs(weight)
    if scale <= 0.0:
        return 0.0
    return float(total / scale)


def _entropy_cost_class(memory_entropy_delta: float, thresholds: Mapping[str, Any]) -> str:
    high = float(thresholds.get("high", 0.20))
    mid = float(thresholds.get("mid", 0.05))
    if memory_entropy_delta >= high:
        return "HIGH"
    if memory_entropy_delta >= mid:
        return "MID"
    return "LOW"


def _resolve_phase_context(
    policy: Mapping[str, Any],
    previous_state: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    profiles = policy.get("phase_profiles") if isinstance(policy.get("phase_profiles"), Mapping) else {}
    default_phase = str(policy.get("default_phase") or "stabilization")
    selected_phase = default_phase
    override_phase = policy.get("memory_phase_override")
    override_applied = False
    if isinstance(override_phase, str) and override_phase in profiles:
        selected_phase = override_phase
        override_applied = True
    elif isinstance(override_phase, str) and override_phase:
        selected_phase = default_phase
    if isinstance(previous_state, Mapping):
        prev_phase = previous_state.get("memory_phase")
        if (
            not (isinstance(override_phase, str) and override_phase in profiles)
            and isinstance(prev_phase, str)
            and prev_phase in profiles
        ):
            selected_phase = prev_phase
    profile = profiles.get(selected_phase) if isinstance(profiles, Mapping) else None
    if not isinstance(profile, Mapping):
        profile = {}
    weights = (
        profile.get("delta_weights")
        if isinstance(profile.get("delta_weights"), Mapping)
        else policy.get("delta_weights") or {}
    )
    thresholds = (
        profile.get("entropy_cost_class_thresholds")
        if isinstance(profile.get("entropy_cost_class_thresholds"), Mapping)
        else policy.get("entropy_cost_class_thresholds") or {}
    )
    return {
        "memory_phase": selected_phase,
        "phase_weight_profile": str(
            profile.get("phase_weight_profile") or f"phase.{selected_phase}.default"
        ),
        "delta_weights": dict(weights),
        "entropy_cost_class_thresholds": dict(thresholds),
        "phase_override_applied": override_applied,
    }


def _value_projection_fingerprint(
    *,
    policy_version: str,
    entropy_model_id: str,
    memory_phase: str,
    phase_weight_profile: str,
    delta_weights: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> str:
    payload = {
        "policy_version": policy_version,
        "entropy_model_id": entropy_model_id,
        "memory_phase": memory_phase,
        "phase_weight_profile": phase_weight_profile,
        "delta_weights": dict(delta_weights),
        "entropy_cost_class_thresholds": dict(thresholds),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _resource_budget_state(
    *,
    record_count: int,
    policy: Mapping[str, Any],
    previous_state: Mapping[str, Any] | None,
) -> Dict[str, Dict[str, float]]:
    cfg = policy.get("resource_budget") if isinstance(policy.get("resource_budget"), Mapping) else {}
    prev_resource = (
        previous_state.get("resource_budget")
        if isinstance(previous_state, Mapping) and isinstance(previous_state.get("resource_budget"), Mapping)
        else {}
    )
    entropy = _channel_state(
        cfg=_mapping(cfg).get("entropy"),
        prev=_mapping(prev_resource).get("entropy"),
        record_count=record_count,
        default_spend_per_record=0.0,
        default_recover_per_night=0.02,
    )
    attention = _channel_state(
        cfg=_mapping(cfg).get("attention"),
        prev=_mapping(prev_resource).get("attention"),
        record_count=record_count,
        default_spend_per_record=0.004,
        default_recover_per_night=0.05,
    )
    affect = _channel_state(
        cfg=_mapping(cfg).get("affect"),
        prev=_mapping(prev_resource).get("affect"),
        record_count=record_count,
        default_spend_per_record=0.003,
        default_recover_per_night=0.04,
    )
    return {
        "entropy": entropy,
        "attention": attention,
        "affect": affect,
    }


def _channel_state(
    *,
    cfg: Any,
    prev: Any,
    record_count: int,
    default_spend_per_record: float,
    default_recover_per_night: float,
) -> Dict[str, float]:
    cfg_map = _mapping(cfg)
    prev_map = _mapping(prev)
    capacity = max(0.0, _safe_float(cfg_map.get("capacity"), 1.0))
    spend_per_record = max(
        0.0,
        _safe_float(cfg_map.get("spend_per_record"), default_spend_per_record),
    )
    recover_per_night = max(
        0.0,
        _safe_float(cfg_map.get("recover_per_night"), default_recover_per_night),
    )
    prev_level = _clamp(_safe_float(prev_map.get("level"), capacity), 0.0, capacity)
    spent = min(prev_level, max(0, record_count) * spend_per_record)
    recovered = recover_per_night
    level = _clamp(prev_level - spent + recovered, 0.0, capacity)
    return {
        "capacity": float(capacity),
        "level": float(level),
        "prev_level": float(prev_level),
        "spent": float(spent),
        "recovered": float(recovered),
    }


def _metabolism_invariants(resource_budget: Mapping[str, Any]) -> Dict[str, Any]:
    channel_details: Dict[str, Any] = {}
    max_error = 0.0
    lower_violations = 0
    upper_violations = 0
    for channel_name in ("entropy", "attention", "affect"):
        node = _mapping(resource_budget.get(channel_name))
        capacity = max(0.0, _safe_float(node.get("capacity"), 0.0))
        prev_level = _safe_float(node.get("prev_level"), capacity)
        spent = _safe_float(node.get("spent"), 0.0)
        recovered = _safe_float(node.get("recovered"), 0.0)
        level = _safe_float(node.get("level"), capacity)
        expected = _clamp(prev_level - spent + recovered, 0.0, capacity)
        error = abs(level - expected)
        max_error = max(max_error, error)
        below = level < -1e-9
        above = level > capacity + 1e-9
        if below:
            lower_violations += 1
        if above:
            upper_violations += 1
        channel_details[channel_name] = {
            "capacity": float(capacity),
            "prev_level": float(prev_level),
            "level": float(level),
            "expected_level": float(expected),
            "conservation_error": float(error),
            "lower_bound_ok": not below,
            "upper_bound_ok": not above,
        }
    return {
        "ok": max_error <= 1e-6 and lower_violations == 0 and upper_violations == 0,
        "max_conservation_error": float(max_error),
        "lower_bound_violations": int(lower_violations),
        "upper_bound_violations": int(upper_violations),
        "channels": channel_details,
    }


def _mapping(candidate: Any) -> Mapping[str, Any]:
    if isinstance(candidate, Mapping):
        return candidate
    return {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(_mapping(out.get(key)), value)
        else:
            out[key] = value
    return out


__all__ = [
    "DEFAULT_METABOLISM_POLICY",
    "load_metabolism_policy",
    "run_metabolism_cycle",
]
