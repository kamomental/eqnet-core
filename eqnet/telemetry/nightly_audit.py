
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple
from collections import Counter

EVIDENCE_DEFAULT_LIMIT = 10
DEFAULT_HEALTH_THRESHOLDS = {
    "fatal_failures_red": 0,
    "warn_fail_rate_yellow": 0.2,
    "boundary_span_yellow": 20,
    "boundary_span_red": 60,
    "prospection_reject_low_yellow": 0.2,
    "prospection_reject_high_yellow": 0.8,
}


@dataclass(frozen=True)
class NightlyAuditConfig:
    trace_root: Path
    out_root: Path
    date_yyyy_mm_dd: str
    boundary_threshold: float = 0.5
    health_thresholds: Dict[str, Any] | None = None
    evidence_limit: int = EVIDENCE_DEFAULT_LIMIT
    memory_reference_log_path: Path | None = None
    think_log_path: Path | None = Path("logs/think_log.jsonl")
    act_log_path: Path | None = Path("logs/act_log.jsonl")


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _get_boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict) and "pass" in value:
        return bool(value["pass"])
    return bool(value)


def _bounded_append(items: List[Dict[str, Any]], data: Dict[str, Any], limit: int) -> None:
    if len(items) < limit:
        items.append(data)


def _short_reasons(reasons: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(reasons, dict):
        return {}
    subset: Dict[str, Any] = {}
    for idx, (key, value) in enumerate(reasons.items()):
        if idx >= 3:
            break
        subset[key] = value
    return subset


def _invariant_evidence(inv_id: str, row: Dict[str, Any], trace_file: str) -> Dict[str, Any]:
    boundary = row.get("boundary") or {}
    snippet = {
        "boundary_score": boundary.get("score"),
        "winner": (row.get("self") or {}).get("winner"),
        "policy_throttles": (row.get("policy") or {}).get("throttles"),
    }
    if boundary.get("reasons"):
        snippet["boundary_reasons"] = _short_reasons(boundary.get("reasons"))
    return {
        "id": inv_id,
        "turn_id": row.get("turn_id"),
        "scenario_id": row.get("scenario_id"),
        "timestamp_ms": row.get("timestamp_ms"),
        "source_loop": row.get("source_loop"),
        "trace_file": trace_file,
        "snippet": snippet,
    }


def _collect_boundary_spans(
    records: List[Dict[str, Any]],
    threshold: float,
    lookup: Dict[int, Tuple[Dict[str, Any], str]],
    limit: int,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    spans: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []
    in_span = False
    start_ts = 0
    last_ts = 0
    length = 0
    last_span_end_index: int | None = None
    current_chain = 0
    max_chain = 0
    last_span_index: int | None = None

    ordered = sorted(records, key=lambda item: int(item.get("timestamp_ms", 0)))
    max_length = 0

    def _span_evidence(ts: int) -> Dict[str, Any]:
        record, trace_file = lookup.get(ts, ({}, ""))
        boundary = record.get("boundary") or {}
        return {
            "start_ts": start_ts,
            "end_ts": last_ts,
            "length": length,
            "start_turn_id": record.get("turn_id"),
            "start_scenario_id": record.get("scenario_id"),
            "trace_file": trace_file,
            "boundary_reasons": _short_reasons(boundary.get("reasons")),
        }

    for idx, record in enumerate(ordered):
        ts = int(record.get("timestamp_ms", 0))
        score = float((record.get("boundary") or {}).get("score", 0.0) or 0.0)
        if score >= threshold:
            if not in_span:
                in_span = True
                start_ts = ts
                length = 0
            length += 1
            last_ts = ts
            last_span_index = idx
        else:
            if in_span:
                spans.append({"start_ts": start_ts, "end_ts": last_ts, "length": length})
                max_length = max(max_length, length)
                _bounded_append(evidence, _span_evidence(start_ts), limit)
                end_index = last_span_index if last_span_index is not None else idx - 1
                if last_span_end_index is None or end_index - last_span_end_index > 1:
                    current_chain = 1
                else:
                    current_chain += 1
                max_chain = max(max_chain, current_chain)
                last_span_end_index = end_index
                in_span = False
    if in_span:
        spans.append({"start_ts": start_ts, "end_ts": last_ts, "length": length})
        max_length = max(max_length, length)
        _bounded_append(evidence, _span_evidence(start_ts), limit)
        end_index = last_span_index if last_span_index is not None else len(ordered) - 1
        if last_span_end_index is None or end_index - last_span_end_index > 1:
            current_chain = 1
        else:
            current_chain += 1
        max_chain = max(max_chain, current_chain)

    summary = {
        "threshold": threshold,
        "span_count": len(spans),
        "max_length": max_length,
        "span_chain_max": max_chain,
        "spans": spans[:50],
    }
    return summary, evidence


def _evaluate_health(
    fatal_failures: int,
    warn_fail_rate: float | None,
    boundary_max: int,
    prospection_reject_rate: float | None,
    thresholds: Dict[str, Any],
) -> Dict[str, Any]:
    status = "GREEN"
    reasons: List[Dict[str, str]] = []
    rank = {"GREEN": 0, "YELLOW": 1, "RED": 2}

    def bump(level: str, message: str) -> None:
        nonlocal status
        if rank[level] > rank[status]:
            status = level
        reasons.append({"level": level, "reason": message})

    if fatal_failures > thresholds.get("fatal_failures_red", 0):
        bump("RED", f"fatal invariant failures: {fatal_failures}")

    if warn_fail_rate is not None and warn_fail_rate > thresholds.get("warn_fail_rate_yellow", 0.0):
        bump("YELLOW", f"warn invariant fail rate {warn_fail_rate:.2f}")

    if boundary_max > thresholds.get("boundary_span_red", 0):
        bump("RED", f"boundary span length {boundary_max}")
    elif boundary_max > thresholds.get("boundary_span_yellow", 0):
        bump("YELLOW", f"boundary span length {boundary_max}")

    if prospection_reject_rate is not None:
        low = thresholds.get("prospection_reject_low_yellow", 0.0)
        high = thresholds.get("prospection_reject_high_yellow", 1.0)
        if prospection_reject_rate < low or prospection_reject_rate > high:
            bump(
                "YELLOW",
                f"prospection reject rate {prospection_reject_rate:.2f} outside [{low:.2f}, {high:.2f}]",
            )

    return {"status": status, "reasons": reasons}


def _bump_health(health: Dict[str, Any], *, level: str, reason: str) -> Dict[str, Any]:
    rank = {"GREEN": 0, "YELLOW": 1, "RED": 2}
    current = str(health.get("status") or "GREEN")
    if rank.get(level, 0) > rank.get(current, 0):
        health["status"] = level
    reasons = health.setdefault("reasons", [])
    if isinstance(reasons, list):
        reasons.append({"level": level, "reason": reason})
    return health


def _ru_v0_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    gate_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    missing_events = 0
    ru_events = 0
    for record in records:
        ru = record.get("ru_v0")
        if not isinstance(ru, dict):
            continue
        ru_events += 1
        gate_action = ru.get("gate_action")
        if gate_action:
            gate_counts[str(gate_action)] += 1
        policy_version = ru.get("policy_version") or record.get("policy_version")
        if policy_version:
            policy_counts[str(policy_version)] += 1
        missing = ru.get("missing_required_fields") or ru.get("missing")
        if isinstance(missing, list):
            if missing:
                missing_events += 1
        elif isinstance(missing, int):
            if missing > 0:
                missing_events += 1
    return {
        "gate_action_counts": dict(gate_counts),
        "policy_version_counts": dict(policy_counts),
        "missing_required_fields_events": missing_events,
        "ru_v0_events": ru_events,
    }


def _qualia_gate_boundary_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    terms: List[float] = []
    ignored_count = 0
    total = 0
    for record in records:
        payload = record.get("qualia_gate")
        if not isinstance(payload, dict):
            continue
        total += 1
        term = payload.get("boundary_term")
        if isinstance(term, (int, float)):
            terms.append(float(term))
        if payload.get("boundary_curve_ignored") is True:
            ignored_count += 1
    if not terms and not ignored_count:
        return {}
    terms_sorted = sorted(terms)

    def _p95(xs: List[float]) -> float:
        if not xs:
            return 0.0
        idx = int((len(xs) - 1) * 0.95)
        return float(xs[min(max(idx, 0), len(xs) - 1)])

    summary: Dict[str, Any] = {}
    if terms_sorted:
        summary["boundary_term"] = {
            "mean": float(sum(terms_sorted) / len(terms_sorted)),
            "p95": _p95(terms_sorted),
            "max": float(terms_sorted[-1]),
            "count": int(len(terms_sorted)),
        }
    if ignored_count:
        summary["boundary_curve_ignored"] = {
            "count": int(ignored_count),
            "ratio": float(ignored_count / total) if total else 0.0,
        }
    return summary


def _qualia_gate_presence_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    suppress = 0
    mode_total = 0
    presence_mode = 0
    for record in records:
        payload = record.get("qualia_gate")
        if not isinstance(payload, dict):
            continue
        if "allow" not in payload:
            continue
        total += 1
        allow = payload.get("allow")
        if allow is False:
            suppress += 1
        mode = record.get("talk_mode")
        if isinstance(mode, str):
            mode_total += 1
            if mode.lower() == "presence":
                presence_mode += 1
    if total == 0:
        return {}
    stats = {
        "ack_silence_ratio": float(suppress / total),
        "ack_silence_count": int(suppress),
        "sample_count": int(total),
    }
    if mode_total > 0:
        stats["presence_mode_ratio"] = float(presence_mode / mode_total)
        stats["presence_mode_count"] = int(presence_mode)
        stats["talk_mode_samples"] = int(mode_total)
    return stats


def _memory_hint_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    hint_events = 0
    shown = 0
    blocked = 0
    verbatim_blocked = 0
    key_counts: Counter[str] = Counter()
    blocked_reason_counts: Counter[str] = Counter()
    interrupt_cost_blocked: List[float] = []
    pressure_values: List[float] = []
    pressure_delta_values: List[float] = []
    pressure_blocked: List[float] = []
    community_turn_violations = 0
    category_blocked_reason: Dict[str, Counter[str]] = {}
    for record in records:
        total += 1
        trace_obs = record.get("trace_observations") or {}
        policy = trace_obs.get("policy") if isinstance(trace_obs, dict) else {}
        hint = policy.get("memory_hint") if isinstance(policy, dict) else None
        if not isinstance(hint, dict):
            continue
        if hint.get("enabled") is not True:
            continue
        hint_events += 1
        if hint.get("shown"):
            shown += 1
            key = hint.get("key")
            if isinstance(key, str) and key:
                key_counts[key] += 1
        pressure = hint.get("pressure")
        if isinstance(pressure, (int, float)):
            pressure_values.append(float(pressure))
            if hint.get("blocked"):
                pressure_blocked.append(float(pressure))
        delta = hint.get("pressure_delta")
        if isinstance(delta, (int, float)):
            pressure_delta_values.append(float(delta))
        if hint.get("blocked"):
            blocked += 1
            reason = hint.get("reason")
            if isinstance(reason, str) and reason:
                blocked_reason_counts[reason] += 1
                key = hint.get("key")
                category = _extract_memory_hint_category(key) if isinstance(key, str) else "unknown"
                bucket = category_blocked_reason.setdefault(category, Counter())
                bucket[reason] += 1
            if reason == "ban_pattern":
                verbatim_blocked += 1
            if hint.get("reason") == "turn_taking" and hint.get("social_mode") == "community":
                community_turn_violations += 1
            cost = hint.get("interrupt_cost")
            if isinstance(cost, (int, float)):
                interrupt_cost_blocked.append(float(cost))
    if total == 0:
        return {}
    category_counts: Counter[str] = Counter()
    for key, count in key_counts.items():
        category_counts[_extract_memory_hint_category(key)] += count
    category_blocked_reason_payload = {
        category: dict(counter)
        for category, counter in sorted(
            category_blocked_reason.items(), key=lambda item: item[0]
        )
    }
    stats = {
        "memory_hint_rate": float(shown / total),
        "memory_hint_blocked_rate": float(blocked / max(1, hint_events)),
        "memory_hint_total": int(hint_events),
        "memory_hint_shown": int(shown),
        "memory_hint_blocked": int(blocked),
        "memory_hint_verbatim_block_count": int(verbatim_blocked),
        "memory_hint_key_topk": key_counts.most_common(5),
        "memory_hint_category_topk": category_counts.most_common(5),
        "memory_hint_category_blocked_reason": category_blocked_reason_payload,
        "memory_hint_blocked_reason_topk": blocked_reason_counts.most_common(5),
        "avg_interrupt_cost_when_blocked": float(sum(interrupt_cost_blocked) / max(1, len(interrupt_cost_blocked))),
        "community_turn_violation_count": int(community_turn_violations),
        "memory_hint_pressure_mean": float(sum(pressure_values) / max(1, len(pressure_values))),
        "memory_hint_pressure_p95": float(sorted(pressure_values)[int(0.95 * (len(pressure_values) - 1))]) if pressure_values else 0.0,
        "memory_hint_pressure_delta_mean": float(sum(pressure_delta_values) / max(1, len(pressure_delta_values))),
        "memory_hint_blocked_pressure_mean": float(sum(pressure_blocked) / max(1, len(pressure_blocked))),
    }
    return stats


def _extract_memory_hint_category(key: str) -> str:
    if not key:
        return "unknown"
    parts = key.split(".")
    if len(parts) >= 3 and parts[0] == "memory_hint":
        return parts[1] or "unknown"
    if len(parts) == 2 and parts[0] == "memory_hint":
        return "legacy"
    return "unknown"


def _closed_loop_trace_coverage(records: List[Dict[str, Any]], day_key: str) -> Dict[str, Any]:
    life_count = 0
    policy_count = 0
    output_count = 0
    linked_count = 0
    first_seen_ts: int | None = None
    last_seen_ts: int | None = None

    for record in records:
        policy_obs = (
            ((record.get("policy") or {}).get("observations") or {}).get("hub") or {}
        )
        qualia_obs = (
            ((record.get("qualia") or {}).get("observations") or {}).get("hub") or {}
        )
        if not isinstance(policy_obs, dict) or not isinstance(qualia_obs, dict):
            continue
        if str(policy_obs.get("day_key") or "") != day_key:
            continue

        ts = record.get("timestamp_ms")
        if isinstance(ts, int):
            first_seen_ts = ts if first_seen_ts is None else min(first_seen_ts, ts)
            last_seen_ts = ts if last_seen_ts is None else max(last_seen_ts, ts)

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

    missing_keys: List[str] = []
    if life_count == 0:
        missing_keys.append("life_indicator_fingerprint")
    if policy_count == 0:
        missing_keys.append("policy_prior_fingerprint")
    if output_count == 0:
        missing_keys.append("output_control_fingerprint")
    if linked_count == 0:
        missing_keys.append("closed_loop_link")

    return {
        "closed_loop_trace_ok": len(missing_keys) == 0,
        "missing_keys": missing_keys,
        "life_indicator_count": life_count,
        "policy_prior_count": policy_count,
        "output_control_count": output_count,
        "linked_count": linked_count,
        "first_seen_ts": first_seen_ts,
        "last_seen_ts": last_seen_ts,
    }


def _iter_memory_reference_for_day(path: Path, day_key: str) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            ts = row.get("ts")
            if not isinstance(ts, (int, float)):
                continue
            utc_day = datetime.fromtimestamp(float(ts), timezone.utc).strftime("%Y-%m-%d")
            if utc_day == day_key:
                yield row


def _recall_cue_budget_audit(path: Path | None, day_key: str) -> Dict[str, Any]:
    if path is None:
        return {
            "recall_cue_ok": True,
            "rarity_budget_ok": True,
            "missing_keys": [],
            "counts": {
                "memory_reference_total": 0,
                "cue_v1_count": 0,
                "suppressed_count": 0,
            },
            "reasons": {},
        }
    total = 0
    cue_v1 = 0
    suppressed = 0
    missing_keys: list[str] = []
    reason_counts: Counter[str] = Counter()
    for row in _iter_memory_reference_for_day(path, day_key):
        total += 1
        mode = row.get("recall_render_mode")
        if isinstance(mode, str) and mode == "cue_v1":
            cue_v1 += 1
        rarity = row.get("rarity_budget")
        if isinstance(rarity, dict):
            if bool(rarity.get("suppressed")):
                suppressed += 1
            reason = rarity.get("reason")
            if isinstance(reason, str) and reason:
                reason_counts[reason] += 1
            if not isinstance(rarity.get("suppressed"), bool):
                missing_keys.append("rarity_budget.suppressed")
        else:
            missing_keys.append("rarity_budget")

    if total > 0 and cue_v1 == 0:
        missing_keys.append("recall_render_mode")

    if total == 0:
        recall_cue_ok = True
        rarity_budget_ok = True
    else:
        recall_cue_ok = cue_v1 > 0
        rarity_budget_ok = ("rarity_budget" not in missing_keys and "rarity_budget.suppressed" not in missing_keys)

    return {
        "recall_cue_ok": recall_cue_ok,
        "rarity_budget_ok": rarity_budget_ok,
        "missing_keys": sorted(set(missing_keys)),
        "counts": {
            "memory_reference_total": total,
            "cue_v1_count": cue_v1,
            "suppressed_count": suppressed,
        },
        "reasons": dict(reason_counts),
    }


def _repair_coverage(records: List[Dict[str, Any]], day_key: str) -> Dict[str, Any]:
    total = 0
    trigger_count = 0
    progressed_count = 0
    next_step_count = 0
    state_counts: Counter[str] = Counter()
    missing_keys: list[str] = []
    for row in records:
        policy_obs = (((row.get("policy") or {}).get("observations") or {}).get("hub") or {})
        if not isinstance(policy_obs, dict):
            continue
        if str(policy_obs.get("day_key") or "") != day_key:
            continue
        total += 1
        before = policy_obs.get("repair_state_before")
        after = policy_obs.get("repair_state_after")
        event = policy_obs.get("repair_event")
        reasons = policy_obs.get("repair_reason_codes")
        finger = policy_obs.get("repair_fingerprint")
        if not isinstance(before, str) or not before:
            missing_keys.append("repair_state_before")
        if not isinstance(after, str) or not after:
            missing_keys.append("repair_state_after")
        if not isinstance(event, str) or not event:
            missing_keys.append("repair_event")
        if not isinstance(reasons, list):
            missing_keys.append("repair_reason_codes")
        if not isinstance(finger, str) or not finger:
            missing_keys.append("repair_fingerprint")
        if isinstance(event, str) and event == "TRIGGER":
            trigger_count += 1
        if isinstance(before, str) and isinstance(after, str) and before != after:
            progressed_count += 1
        if isinstance(after, str):
            state_counts[after] += 1
            if after == "NEXT_STEP":
                next_step_count += 1
    stuck_suspected = False
    if trigger_count > 0 and progressed_count == 0 and next_step_count == 0:
        stuck_suspected = True
    return {
        "repair_events_total": total,
        "trigger_count": trigger_count,
        "state_counts": dict(state_counts),
        "progressed_count": progressed_count,
        "next_step_count": next_step_count,
        "stuck_suspected": stuck_suspected,
        "missing_keys": sorted(set(missing_keys)),
    }


def _memory_thermo_contract_coverage(records: List[Dict[str, Any]], day_key: str) -> Dict[str, Any]:
    total = 0
    missing_keys: list[str] = []
    throttle_inconsistency_count = 0
    throttle_reason_missing_count = 0
    throttle_profile_missing_count = 0
    irreversible_without_trace_count = 0
    entropy_class_inconsistency_count = 0
    defrag_metrics_missing_count = 0
    defrag_delta_inconsistency_count = 0
    required = (
        "memory_entropy_delta",
        "memory_phase",
        "phase_weight_profile",
        "value_projection_fingerprint",
        "energy_budget_used",
        "budget_throttle_applied",
        "policy_version",
        "entropy_model_id",
    )
    phase_transition_fp_stale_count = 0
    phase_override_applied_count = 0
    ordered_phase_rows: list[tuple[int, str, str]] = []
    for row in records:
        policy_obs = (((row.get("policy") or {}).get("observations") or {}).get("hub") or {})
        if not isinstance(policy_obs, dict):
            continue
        if str(policy_obs.get("day_key") or "") != day_key:
            continue
        total += 1
        for key in required:
            if key not in policy_obs:
                missing_keys.append(key)
        if "memory_entropy_delta" in policy_obs and not isinstance(policy_obs.get("memory_entropy_delta"), (int, float)):
            missing_keys.append("memory_entropy_delta")
        if "memory_phase" in policy_obs and not isinstance(policy_obs.get("memory_phase"), str):
            missing_keys.append("memory_phase")
        if "energy_budget_used" in policy_obs and not isinstance(policy_obs.get("energy_budget_used"), (int, float)):
            missing_keys.append("energy_budget_used")
        if "budget_throttle_applied" in policy_obs and not isinstance(policy_obs.get("budget_throttle_applied"), bool):
            missing_keys.append("budget_throttle_applied")

        throttle = policy_obs.get("budget_throttle_applied")
        used = policy_obs.get("energy_budget_used")
        limit = policy_obs.get("energy_budget_limit")
        if isinstance(throttle, bool) and throttle and isinstance(used, (int, float)) and isinstance(limit, (int, float)):
            if float(used) < float(limit):
                throttle_inconsistency_count += 1
            reason = policy_obs.get("throttle_reason_code")
            profile = policy_obs.get("output_control_profile")
            if not (isinstance(reason, str) and reason.strip()):
                throttle_reason_missing_count += 1
            if not (isinstance(profile, str) and profile.strip()):
                throttle_profile_missing_count += 1
        irreversible_op = policy_obs.get("irreversible_op")
        if isinstance(irreversible_op, bool) and irreversible_op:
            event_id = policy_obs.get("event_id") or row.get("event_id")
            trace_id = policy_obs.get("trace_id") or row.get("trace_id")
            if not (isinstance(event_id, str) and event_id) or not (isinstance(trace_id, str) and trace_id):
                irreversible_without_trace_count += 1
        entropy_class = policy_obs.get("entropy_cost_class")
        if isinstance(irreversible_op, bool) and irreversible_op and isinstance(entropy_class, str):
            if entropy_class.strip().upper() == "LOW":
                entropy_class_inconsistency_count += 1
        phase = policy_obs.get("memory_phase")
        projection_fp = policy_obs.get("value_projection_fingerprint")
        ts = row.get("timestamp_ms")
        if isinstance(phase, str) and phase and isinstance(projection_fp, str) and projection_fp and isinstance(ts, int):
            ordered_phase_rows.append((ts, phase, projection_fp))
        if bool(policy_obs.get("phase_override_applied")):
            phase_override_applied_count += 1
        before_metrics = policy_obs.get("defrag_metrics_before")
        after_metrics = policy_obs.get("defrag_metrics_after")
        delta_metrics = policy_obs.get("defrag_metrics_delta")
        if isinstance(irreversible_op, bool) and irreversible_op:
            if not isinstance(before_metrics, dict) or not isinstance(after_metrics, dict) or not isinstance(delta_metrics, dict):
                defrag_metrics_missing_count += 1
        if isinstance(before_metrics, dict) and isinstance(after_metrics, dict) and isinstance(delta_metrics, dict):
            keys = set(before_metrics.keys()) | set(after_metrics.keys()) | set(delta_metrics.keys())
            inconsistent = False
            for key in keys:
                b = before_metrics.get(key)
                a = after_metrics.get(key)
                d = delta_metrics.get(key)
                if not isinstance(b, (int, float)) or not isinstance(a, (int, float)) or not isinstance(d, (int, float)):
                    continue
                if abs((float(a) - float(b)) - float(d)) > 1e-6:
                    inconsistent = True
                    break
            if inconsistent:
                defrag_delta_inconsistency_count += 1

    if total == 0:
        return {
            "memory_thermo_contract_ok": True,
            "missing_keys": [],
            "events_checked": 0,
            "throttle_inconsistency_count": 0,
            "throttle_reason_missing_count": 0,
            "throttle_profile_missing_count": 0,
            "irreversible_without_trace_count": 0,
            "entropy_class_inconsistency_count": 0,
            "defrag_metrics_missing_count": 0,
            "defrag_delta_inconsistency_count": 0,
            "phase_transition_fp_stale_count": 0,
            "phase_override_applied_count": 0,
            "warnings": [],
        }
    prev_phase: str | None = None
    prev_fp: str | None = None
    for _ts, phase, fp in sorted(ordered_phase_rows, key=lambda item: item[0]):
        if prev_phase is not None and phase != prev_phase and prev_fp is not None and fp == prev_fp:
            phase_transition_fp_stale_count += 1
        prev_phase = phase
        prev_fp = fp
    return {
        "memory_thermo_contract_ok": (
            len(missing_keys) == 0
            and throttle_inconsistency_count == 0
            and throttle_reason_missing_count == 0
            and throttle_profile_missing_count == 0
            and irreversible_without_trace_count == 0
            and entropy_class_inconsistency_count == 0
            and defrag_metrics_missing_count == 0
            and defrag_delta_inconsistency_count == 0
            and phase_transition_fp_stale_count == 0
        ),
        "missing_keys": sorted(set(missing_keys)),
        "events_checked": total,
        "throttle_inconsistency_count": throttle_inconsistency_count,
        "throttle_reason_missing_count": throttle_reason_missing_count,
        "throttle_profile_missing_count": throttle_profile_missing_count,
        "irreversible_without_trace_count": irreversible_without_trace_count,
        "entropy_class_inconsistency_count": entropy_class_inconsistency_count,
        "defrag_metrics_missing_count": defrag_metrics_missing_count,
        "defrag_delta_inconsistency_count": defrag_delta_inconsistency_count,
        "phase_transition_fp_stale_count": phase_transition_fp_stale_count,
        "phase_override_applied_count": phase_override_applied_count,
        "warnings": (
            ["PHASE_OVERRIDE_APPLIED"]
            if phase_override_applied_count > 0
            else []
        ),
    }


def _separation_governance_audit(
    *,
    think_log_path: Path | None,
    act_log_path: Path | None,
) -> Dict[str, Any]:
    """Assess wiring status of thought/experience separation governance.

    This block is intentionally independent from health scoring.
    """

    def _connected(path: Path | None) -> bool:
        if path is None:
            return False
        if path.exists():
            return True
        return path.parent.exists()

    def _iter_rows(path: Path | None) -> Iterator[Dict[str, Any]]:
        if path is None or not path.exists():
            return iter(())
        return _iter_jsonl(path)

    think_connected = _connected(think_log_path)
    act_connected = _connected(act_log_path)

    # Keep allowed set explicit to avoid importing runtime module dependencies.
    allowed = {
        "experience",
        "imagery",
        "hypothesis",
        "borrowed_idea",
        "discussion",
        "unknown",
    }

    memory_kind_rows = 0
    memory_kind_bad = 0
    promotion_events = 0
    source_misattribution_events = 0
    for row in _iter_rows(think_log_path):
        kind = row.get("memory_kind")
        if isinstance(kind, str):
            memory_kind_rows += 1
            if kind not in allowed:
                memory_kind_bad += 1
        meta = row.get("meta")
        if isinstance(meta, dict):
            ev = meta.get("audit_event")
            if isinstance(ev, str) and ev in {"PROMOTION_GUARD_BLOCKED", "PROMOTION_GUARD_PASSED"}:
                promotion_events += 1
            if isinstance(ev, str) and ev in {"SOURCE_FUZZY", "DOUBLE_TAKE"}:
                source_misattribution_events += 1
        ev_top = row.get("audit_event")
        if isinstance(ev_top, str) and ev_top in {"SOURCE_FUZZY", "DOUBLE_TAKE"}:
            source_misattribution_events += 1
    for row in _iter_rows(act_log_path):
        kind = row.get("memory_kind")
        if isinstance(kind, str):
            memory_kind_rows += 1
            if kind not in allowed:
                memory_kind_bad += 1
        meta = row.get("meta")
        if isinstance(meta, dict) and isinstance(meta.get("promotion_evidence_event_id"), str):
            promotion_events += 1

    memory_kind_enforced = memory_kind_rows > 0 and memory_kind_bad == 0
    promotion_guard_enforced = promotion_events > 0
    source_misattribution_wired = think_log_path is not None
    source_misattribution_active = source_misattribution_events > 0

    checks = {
        "think_log_connected": think_connected,
        "act_log_connected": act_connected,
        "memory_kind_enum_enforced": memory_kind_enforced,
        "promotion_guard_enforced": promotion_guard_enforced,
        "source_misattribution_events": {
            "wired": source_misattribution_wired,
            "active": source_misattribution_active,
            "active_count": int(source_misattribution_events),
        },
        # Backward-compat key kept for existing consumers.
        "source_misattribution_events_connected": source_misattribution_active,
    }

    core_ready = (
        checks["think_log_connected"]
        and checks["act_log_connected"]
        and checks["memory_kind_enum_enforced"]
        and checks["promotion_guard_enforced"]
    )
    partial_ready = (
        checks["think_log_connected"]
        or checks["act_log_connected"]
        or checks["memory_kind_enum_enforced"]
        or checks["promotion_guard_enforced"]
    )

    if core_ready:
        status = "INSTALLED_BASELINE"
        enabled = True
        reason = "thought_experience_separation_baseline_connected"
    elif partial_ready:
        status = "PARTIALLY_INSTALLED"
        enabled = False
        reason = "thought_experience_separation_partially_connected"
    else:
        status = "NOT_INSTALLED"
        enabled = False
        reason = "thought_experience_separation_sensors_not_connected"

    if status != "NOT_INSTALLED" and source_misattribution_wired:
        status += "+SOURCE_RECHECK_WIRED"
    if status != "NOT_INSTALLED" and source_misattribution_active:
        status += "+SOURCE_RECHECK_ACTIVE"

    return {
        "enabled": enabled,
        "status": status,
        "reason": reason,
        "checks": checks,
    }


def generate_audit(cfg: NightlyAuditConfig) -> Path:
    day_dir = cfg.trace_root / cfg.date_yyyy_mm_dd
    if not day_dir.exists():
        raise FileNotFoundError(day_dir)

    files = sorted(day_dir.glob("*.jsonl"))
    records: List[Dict[str, Any]] = []
    record_lookup: Dict[int, Tuple[Dict[str, Any], str]] = {}
    invariant_total: Dict[str, int] = {}
    invariant_pass: Dict[str, int] = {}
    fatal_failures = 0
    warn_checks = 0
    warn_failures = 0
    fatal_evidence: List[Dict[str, Any]] = []
    warn_evidence: List[Dict[str, Any]] = []
    prospection_total = 0
    prospection_reject = 0
    prospection_samples: List[Dict[str, Any]] = []
    throttles: Dict[str, int] = {}
    throttle_examples: Dict[str, List[Dict[str, Any]]] = {}
    evidence_limit = cfg.evidence_limit or EVIDENCE_DEFAULT_LIMIT

    def record_invariant(
        severity: str,
        inv_id: str,
        passed: Any,
        row: Dict[str, Any],
        trace_file: str,
    ) -> None:
        nonlocal fatal_failures, warn_checks, warn_failures
        ok = _get_boolish(passed)
        invariant_total[inv_id] = invariant_total.get(inv_id, 0) + 1
        if ok:
            invariant_pass[inv_id] = invariant_pass.get(inv_id, 0) + 1
        if severity == "fatal":
            if not ok:
                fatal_failures += 1
                _bounded_append(fatal_evidence, _invariant_evidence(inv_id, row, trace_file), evidence_limit)
        else:
            warn_checks += 1
            if not ok:
                warn_failures += 1
                _bounded_append(warn_evidence, _invariant_evidence(inv_id, row, trace_file), evidence_limit)

    for fp in files:
        trace_file = str(fp)
        for row in _iter_jsonl(fp):
            records.append(row)
            ts = int(row.get("timestamp_ms", 0))
            record_lookup.setdefault(ts, (row, trace_file))

            inv_payload = row.get("invariants") or {}
            if isinstance(inv_payload, dict) and ("fatal" in inv_payload or "warn" in inv_payload):
                for severity_key in ("fatal", "warn"):
                    bucket = inv_payload.get(severity_key)
                    if isinstance(bucket, dict):
                        for inv_id, passed in bucket.items():
                            record_invariant(severity_key, str(inv_id), passed, row, trace_file)
            elif isinstance(inv_payload, dict):
                for inv_id, passed in inv_payload.items():
                    record_invariant("warn", str(inv_id), passed, row, trace_file)

            prospection = row.get("prospection") or {}
            if "accepted" in prospection:
                prospection_total += 1
                if not bool(prospection.get("accepted")):
                    prospection_reject += 1
            sample = {
                "turn_id": row.get("turn_id"),
                "scenario_id": row.get("scenario_id"),
                "timestamp_ms": row.get("timestamp_ms"),
                "jerk": prospection.get("jerk"),
                "temperature": prospection.get("temperature"),
                "accepted": prospection.get("accepted"),
                "trace_file": trace_file,
            }
            if sample["jerk"] is not None or sample["temperature"] is not None:
                prospection_samples.append(sample)

            throttles_payload = (row.get("policy") or {}).get("throttles") or {}
            if isinstance(throttles_payload, dict):
                for key, value in throttles_payload.items():
                    if _get_boolish(value):
                        throttles[key] = throttles.get(key, 0) + 1
                        bucket = throttle_examples.setdefault(key, [])
                        _bounded_append(
                            bucket,
                            {
                                "turn_id": row.get("turn_id"),
                                "scenario_id": row.get("scenario_id"),
                                "timestamp_ms": row.get("timestamp_ms"),
                                "trace_file": trace_file,
                            },
                            evidence_limit,
                        )

    boundary_summary, boundary_evidence = _collect_boundary_spans(
        records,
        cfg.boundary_threshold,
        record_lookup,
        evidence_limit,
    )

    invariants_summary = {
        inv: {
            "total": invariant_total.get(inv, 0),
            "pass": invariant_pass.get(inv, 0),
            "pass_rate": (
                invariant_pass.get(inv, 0) / invariant_total[inv]
                if invariant_total.get(inv)
                else None
            ),
        }
        for inv in sorted(invariant_total.keys())
    }

    warn_fail_rate = (warn_failures / warn_checks) if warn_checks else None
    reject_rate = (prospection_reject / prospection_total) if prospection_total else None
    accept_rate = (1 - reject_rate) if reject_rate is not None else None

    top_jerk = [s for s in prospection_samples if s.get("jerk") is not None]
    top_jerk.sort(key=lambda item: float(item["jerk"]), reverse=True)
    top_jerk = top_jerk[:evidence_limit]

    top_temperature = [s for s in prospection_samples if s.get("temperature") is not None]
    top_temperature.sort(key=lambda item: float(item["temperature"]), reverse=True)
    top_temperature = top_temperature[:evidence_limit]

    evidence = {
        "invariants": {
            "fatal_failures": fatal_evidence[:evidence_limit],
            "warn_failures": warn_evidence[:evidence_limit],
        },
        "boundary_spans": boundary_evidence,
        "prospection": {
            "top_jerk": top_jerk,
            "top_temperature": top_temperature,
        },
        "policy": {
            "throttle_examples": {k: v[:evidence_limit] for k, v in throttle_examples.items()},
        },
    }

    thresholds = dict(DEFAULT_HEALTH_THRESHOLDS)
    if cfg.health_thresholds:
        thresholds.update(cfg.health_thresholds)

    health = _evaluate_health(
        len(fatal_evidence),
        warn_fail_rate,
        boundary_summary.get("max_length", 0) or 0,
        reject_rate,
        thresholds,
    )
    closed_loop = _closed_loop_trace_coverage(records, cfg.date_yyyy_mm_dd)
    if not closed_loop.get("closed_loop_trace_ok", False):
        health = _bump_health(
            health,
            level="YELLOW",
            reason=f"closed-loop trace coverage missing keys: {closed_loop.get('missing_keys')}",
        )
    recall_cue_budget = _recall_cue_budget_audit(
        cfg.memory_reference_log_path,
        cfg.date_yyyy_mm_dd,
    )
    if not recall_cue_budget.get("recall_cue_ok", True):
        health = _bump_health(
            health,
            level="YELLOW",
            reason=f"recall cue coverage missing keys: {recall_cue_budget.get('missing_keys')}",
        )
    if not recall_cue_budget.get("rarity_budget_ok", True):
        health = _bump_health(
            health,
            level="YELLOW",
            reason=f"rarity budget metadata missing keys: {recall_cue_budget.get('missing_keys')}",
        )
    repair_coverage = _repair_coverage(records, cfg.date_yyyy_mm_dd)
    if repair_coverage.get("missing_keys"):
        health = _bump_health(
            health,
            level="YELLOW",
            reason=f"repair coverage missing keys: {repair_coverage.get('missing_keys')}",
        )
    if repair_coverage.get("stuck_suspected"):
        health = _bump_health(
            health,
            level="YELLOW",
            reason="repair trigger detected without progression",
        )
    memory_thermo_contract = _memory_thermo_contract_coverage(records, cfg.date_yyyy_mm_dd)
    if memory_thermo_contract.get("missing_keys"):
        health = _bump_health(
            health,
            level="YELLOW",
            reason=f"memory thermo contract missing keys: {memory_thermo_contract.get('missing_keys')}",
        )
    if int(memory_thermo_contract.get("throttle_inconsistency_count") or 0) > 0:
        health = _bump_health(
            health,
            level="YELLOW",
            reason="budget throttle inconsistency detected",
        )
    if int(memory_thermo_contract.get("throttle_reason_missing_count") or 0) > 0:
        health = _bump_health(
            health,
            level="YELLOW",
            reason="budget throttle reason missing",
        )
    if int(memory_thermo_contract.get("throttle_profile_missing_count") or 0) > 0:
        health = _bump_health(
            health,
            level="YELLOW",
            reason="budget throttle profile missing",
        )
    if int(memory_thermo_contract.get("irreversible_without_trace_count") or 0) > 0:
        health = _bump_health(
            health,
            level="YELLOW",
            reason="irreversible_without_trace detected",
        )
    if int(memory_thermo_contract.get("entropy_class_inconsistency_count") or 0) > 0:
        health = _bump_health(
            health,
            level="YELLOW",
            reason="irreversible_op with LOW entropy_cost_class detected",
        )
    if int(memory_thermo_contract.get("defrag_metrics_missing_count") or 0) > 0:
        health = _bump_health(
            health,
            level="YELLOW",
            reason="irreversible defrag without metrics detected",
        )
    if int(memory_thermo_contract.get("defrag_delta_inconsistency_count") or 0) > 0:
        health = _bump_health(
            health,
            level="YELLOW",
            reason="defrag metrics delta inconsistency detected",
        )
    if int(memory_thermo_contract.get("phase_transition_fp_stale_count") or 0) > 0:
        health = _bump_health(
            health,
            level="YELLOW",
            reason="phase transition without projection fingerprint update detected",
        )

    qualia_gate_stats = _qualia_gate_boundary_stats(records)
    presence_stats = _qualia_gate_presence_stats(records)
    if presence_stats:
        if not qualia_gate_stats:
            qualia_gate_stats = {}
        qualia_gate_stats["presence"] = presence_stats
    memory_hint_stats = _memory_hint_stats(records)
    if memory_hint_stats:
        if not qualia_gate_stats:
            qualia_gate_stats = {}
        qualia_gate_stats["memory_hint"] = memory_hint_stats

    audit = {
        "schema_version": "nightly_audit_v1",
        "date": cfg.date_yyyy_mm_dd,
        "trace_files": [str(fp) for fp in files],
        "invariants": {
            "summary": invariants_summary,
            "fatal_failures": len(fatal_evidence),
            "warn_fail_rate": warn_fail_rate,
        },
        "boundary": boundary_summary,
        "prospection": {
            "total": prospection_total,
            "reject": prospection_reject,
            "reject_rate": reject_rate,
            "accept_rate": accept_rate,
            "top_jerk": top_jerk,
            "top_temperature": top_temperature,
        },
        "policy": {
            "offer_throttle_counts": throttles,
            "offer_throttle_examples": {k: v[:evidence_limit] for k, v in throttle_examples.items()},
        },
        "evidence": evidence,
        "health": health,
        "closed_loop_trace": closed_loop,
        "recall_cue_budget": recall_cue_budget,
        "repair_coverage": repair_coverage,
        "memory_thermo_contract": memory_thermo_contract,
        "separation": _separation_governance_audit(
            think_log_path=cfg.think_log_path,
            act_log_path=cfg.act_log_path,
        ),
        "ru_v0_summary": _ru_v0_summary(records),
        "qualia_gate_stats": qualia_gate_stats,
    }

    cfg.out_root.mkdir(parents=True, exist_ok=True)
    out_path = cfg.out_root / f"nightly_audit_{cfg.date_yyyy_mm_dd}.json"
    out_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


__all__ = ["NightlyAuditConfig", "generate_audit"]
