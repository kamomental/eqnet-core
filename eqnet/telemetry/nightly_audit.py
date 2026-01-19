
from __future__ import annotations

import json
from dataclasses import dataclass
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
        "ru_v0_summary": _ru_v0_summary(records),
        "qualia_gate_stats": qualia_gate_stats,
    }

    cfg.out_root.mkdir(parents=True, exist_ok=True)
    out_path = cfg.out_root / f"nightly_audit_{cfg.date_yyyy_mm_dd}.json"
    out_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


__all__ = ["NightlyAuditConfig", "generate_audit"]
