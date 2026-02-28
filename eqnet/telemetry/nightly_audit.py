
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple
from collections import Counter
from eqnet.runtime.online_delta_promotion import (
    append_rule_delta_promotions,
    decide_promotions,
    load_promotion_policy,
    summarize_online_delta_effectiveness,
)

EVIDENCE_DEFAULT_LIMIT = 10
DEFAULT_HEALTH_THRESHOLDS = {
    "fatal_failures_red": 0,
    "warn_fail_rate_yellow": 0.2,
    "boundary_span_yellow": 20,
    "boundary_span_red": 60,
    "prospection_reject_low_yellow": 0.2,
    "prospection_reject_high_yellow": 0.8,
    "online_delta_block_rate_yellow": 0.8,
    "forced_human_confirm_count_yellow": 20,
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
    state_dir: Path | None = Path("state")
    promotion_policy_path: Path | None = Path("configs/online_delta_promotion_v0.yaml")
    rule_delta_path: Path | None = None
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


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * pct))
    idx = min(max(idx, 0), len(ordered) - 1)
    return float(ordered[idx])


def _extract_lazy_rag_sat_ratio(row: Dict[str, Any]) -> float | None:
    candidates: List[Any] = []
    response_meta = row.get("response_meta")
    if isinstance(response_meta, dict):
        safety = response_meta.get("safety")
        if isinstance(safety, dict):
            candidates.append(safety.get("lazy_rag_sat_ratio"))
    safety_root = row.get("safety")
    if isinstance(safety_root, dict):
        candidates.append(safety_root.get("lazy_rag_sat_ratio"))
    lazy_rag = row.get("lazy_rag")
    if isinstance(lazy_rag, dict):
        candidates.append(lazy_rag.get("sat_ratio"))
    for value in candidates:
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _extract_uncertainty_reasons(row: Dict[str, Any]) -> List[str]:
    candidates: List[Any] = []
    response_meta = row.get("response_meta")
    if isinstance(response_meta, dict):
        candidates.append(response_meta.get("uncertainty_reason"))
        safety = response_meta.get("safety")
        if isinstance(safety, dict):
            candidates.append(safety.get("uncertainty_reason"))
    safety_root = row.get("safety")
    if isinstance(safety_root, dict):
        candidates.append(safety_root.get("uncertainty_reason"))
    reasons: List[str] = []
    for item in candidates:
        if isinstance(item, list):
            for value in item:
                if isinstance(value, str) and value.strip():
                    reasons.append(value)
            if reasons:
                break
        elif isinstance(item, str) and item.strip():
            reasons.append(item)
            break
    return reasons


def _extract_confidence_value(row: Dict[str, Any]) -> float | None:
    candidates: List[Any] = []
    response_meta = row.get("response_meta")
    if isinstance(response_meta, dict):
        candidates.append(response_meta.get("confidence"))
        safety = response_meta.get("safety")
        if isinstance(safety, dict):
            candidates.append(safety.get("confidence"))
    safety_root = row.get("safety")
    if isinstance(safety_root, dict):
        candidates.append(safety_root.get("confidence"))
    for value in candidates:
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _confidence_label(value: float, *, low_max: float, mid_max: float) -> str:
    if value <= low_max:
        return "low"
    if value <= mid_max:
        return "mid"
    return "high"


def _is_sha256_hex(value: Any, *, allow_empty: bool = False) -> bool:
    if value is None:
        return allow_empty
    text = str(value).strip().lower()
    if not text:
        return allow_empty
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _turn_index(turn_id: Any) -> int | None:
    text = str(turn_id or "")
    if not text:
        return None
    match = re.search(r"(\d+)(?!.*\d)", text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _audit_mecpe(cfg: NightlyAuditConfig) -> Dict[str, Any]:
    day_compact = cfg.date_yyyy_mm_dd.replace("-", "")
    candidates = [
        cfg.trace_root.parent / f"mecpe-{day_compact}.jsonl",
        cfg.trace_root / f"mecpe-{day_compact}.jsonl",
    ]
    log_files = [str(path) for path in candidates if path.exists()]
    records: List[Dict[str, Any]] = []
    mecpe_lines_total = 0
    contract_error_total = 0
    contract_error_types: Counter[str] = Counter()
    required_keys = (
        "schema_version",
        "timestamp_ms",
        "turn_id",
        "prompt_hash",
        "model",
        "text_hash",
        "audio_sha256",
        "video_sha256",
    )
    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                mecpe_lines_total += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    contract_error_total += 1
                    contract_error_types["json_decode"] += 1
                    continue
                if not isinstance(row, dict):
                    contract_error_total += 1
                    contract_error_types["invalid_row_type"] += 1
                    continue
                missing_required = [key for key in required_keys if key not in row]
                if missing_required:
                    contract_error_total += 1
                    contract_error_types["missing_required_key"] += 1
                    continue
                if not isinstance(row.get("model"), dict) or not str((row.get("model") or {}).get("version") or "").strip():
                    contract_error_total += 1
                    contract_error_types["invalid_model_version"] += 1
                    continue
                records.append(row)

    total = len(records)
    rationale_hash_count = 0
    missing_rationale_count = 0
    evidence_missing_count = 0
    missing_audio_count = 0
    missing_video_count = 0
    pair_total = 0
    future_conflict_count = 0
    future_conflict_samples: List[str] = []
    span_missing_count = 0
    hash_integrity_ok_count = 0
    hash_invalid_count = 0

    for row in records:
        stage1 = row.get("stage1_emotion") if isinstance(row.get("stage1_emotion"), dict) else {}
        stage2 = row.get("stage2_cause_pair") if isinstance(row.get("stage2_cause_pair"), dict) else {}
        stage3 = row.get("stage3_cause_span") if isinstance(row.get("stage3_cause_span"), dict) else {}

        stage1_rationale = str(stage1.get("rationale_hash") or "").strip()
        stage2_rationale = str(stage2.get("rationale_hash") or "").strip()
        if stage1_rationale or stage2_rationale:
            rationale_hash_count += 1
        else:
            missing_rationale_count += 1

        audio_sha = str(row.get("audio_sha256") or "").strip()
        video_sha = str(row.get("video_sha256") or "").strip()
        if not audio_sha:
            missing_audio_count += 1
        if not video_sha:
            missing_video_count += 1
        if not audio_sha or not video_sha:
            evidence_missing_count += 1

        has_pair = bool(stage2.get("cause_turn_id"))
        if has_pair:
            pair_total += 1
            target_index = _turn_index(row.get("turn_id"))
            cause_index = _turn_index(stage2.get("cause_turn_id"))
            if target_index is not None and cause_index is not None and cause_index > target_index:
                future_conflict_count += 1
                if len(future_conflict_samples) < 3:
                    future_conflict_samples.append(str(row.get("turn_id") or ""))
            has_valid_span = isinstance(stage3.get("start_char"), int) and isinstance(stage3.get("end_char"), int)
            if not has_valid_span:
                span_missing_count += 1

        schema_ok = row.get("schema_version") == "mecpe_record.v0"
        model = row.get("model") if isinstance(row.get("model"), dict) else {}
        model_ok = bool(str(model.get("version") or "").strip())
        hash_ok = (
            _is_sha256_hex(row.get("prompt_hash"))
            and _is_sha256_hex(row.get("text_hash"))
            and _is_sha256_hex(row.get("audio_sha256"), allow_empty=True)
            and _is_sha256_hex(row.get("video_sha256"), allow_empty=True)
        )
        if schema_ok and model_ok and hash_ok:
            hash_integrity_ok_count += 1
        else:
            hash_invalid_count += 1
            if (
                not _is_sha256_hex(row.get("prompt_hash"))
                or not _is_sha256_hex(row.get("text_hash"))
                or not _is_sha256_hex(row.get("audio_sha256"), allow_empty=True)
                or not _is_sha256_hex(row.get("video_sha256"), allow_empty=True)
            ):
                contract_error_total += 1
                contract_error_types["invalid_hash_len"] += 1

    denom = float(total) if total else 0.0
    pair_denom = float(pair_total) if pair_total else 0.0
    line_denom = float(mecpe_lines_total) if mecpe_lines_total else 0.0
    top_type = ""
    if contract_error_types:
        top_type = contract_error_types.most_common(1)[0][0]
    return {
        "log_files": log_files,
        "mecpe_lines_total": mecpe_lines_total,
        "total_records": total,
        "rationale_hash_coverage": round((rationale_hash_count / denom), 3) if denom else 0.0,
        "missing_rationale_count": missing_rationale_count,
        "evidence_staleness": {
            "missing_count": evidence_missing_count,
            "missing_audio_count": missing_audio_count,
            "missing_video_count": missing_video_count,
            "missing_ratio": round((evidence_missing_count / denom), 3) if denom else 0.0,
        },
        "future_cause_conflict": {
            "pair_total": pair_total,
            "conflict_count": future_conflict_count,
            "conflict_rate": round((future_conflict_count / pair_denom), 3) if pair_denom else 0.0,
            "sample_ids": future_conflict_samples,
        },
        "span_missing": {
            "pair_total": pair_total,
            "missing_count": span_missing_count,
            "missing_rate": round((span_missing_count / pair_denom), 3) if pair_denom else 0.0,
        },
        "hash_integrity": {
            "ok_count": hash_integrity_ok_count,
            "invalid_count": hash_invalid_count,
            "ok_rate": round((hash_integrity_ok_count / denom), 3) if denom else 0.0,
        },
        "contract_errors": {
            "total": contract_error_total,
            "ratio": round((contract_error_total / line_denom), 3) if line_denom else 0.0,
            "top_type": top_type,
            "by_type": dict(contract_error_types),
        },
    }


def _ru_v0_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    gate_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    blocked_tool_names: Counter[str] = Counter()
    blocked_reason_codes: Counter[str] = Counter()
    forced_gate_actions: Counter[str] = Counter()
    missing_events = 0
    ru_events = 0
    tool_call_attempt_count = 0
    tool_call_blocked_count = 0
    forced_human_confirm_count = 0
    online_delta_applied_count = 0
    online_delta_missing_contract_count = 0
    for record in records:
        event_type = str(record.get("event_type") or "")
        forced_in_record = False
        if event_type == "tool_call":
            tool_call_attempt_count += 1
        if event_type == "tool_call_blocked":
            tool_call_attempt_count += 1
            tool_call_blocked_count += 1
            tool_name = str(record.get("tool_name") or "")
            if tool_name:
                blocked_tool_names[tool_name] += 1
            for code in (record.get("reason_codes") or []):
                if isinstance(code, str) and code:
                    blocked_reason_codes[code] += 1
        if event_type == "forced_gate_action":
            forced_value = str(record.get("forced_gate_action") or "")
            if forced_value:
                forced_gate_actions[forced_value] += 1
                if forced_value == "HUMAN_CONFIRM":
                    forced_in_record = True
        if event_type == "online_delta_discarded":
            for code in (record.get("reason_codes") or []):
                if isinstance(code, str) and code.startswith("MISSING_REQUIRED_KEY"):
                    online_delta_missing_contract_count += 1
                    break

        policy_obs = (
            ((record.get("policy") or {}).get("observations") or {}).get("hub")
            if isinstance(record.get("policy"), dict)
            else None
        )
        if isinstance(policy_obs, dict):
            if bool(policy_obs.get("online_delta_applied")):
                online_delta_applied_count += 1
            forced_value = str(policy_obs.get("forced_gate_action") or "")
            if forced_value:
                forced_gate_actions[forced_value] += 1
                if forced_value == "HUMAN_CONFIRM":
                    forced_in_record = True
        if forced_in_record:
            forced_human_confirm_count += 1

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
    block_rate = (
        (tool_call_blocked_count / tool_call_attempt_count)
        if tool_call_attempt_count > 0
        else 0.0
    )
    return {
        "gate_action_counts": dict(gate_counts),
        "policy_version_counts": dict(policy_counts),
        "missing_required_fields_events": missing_events,
        "ru_v0_events": ru_events,
        "tool_call_attempt_count": tool_call_attempt_count,
        "tool_call_blocked_count": tool_call_blocked_count,
        "forced_human_confirm_count": forced_human_confirm_count,
        "online_delta_applied_count": online_delta_applied_count,
        "online_delta_missing_contract_count": online_delta_missing_contract_count,
        "blocked_tool_names_topk": blocked_tool_names.most_common(5),
        "blocked_reason_codes_topk": blocked_reason_codes.most_common(5),
        "forced_gate_action_topk": forced_gate_actions.most_common(5),
        "online_delta_effectiveness": {
            "tool_block_rate": round(block_rate, 3),
        },
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
    metabolism_invariants_failed_count = 0
    transaction_missing_count = 0
    transaction_non_atomic_count = 0
    repair_plan_missing_count = 0
    repair_ops_digest_missing_count = 0
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
        inv_ok = policy_obs.get("metabolism_invariants_ok")
        if isinstance(inv_ok, bool) and not inv_ok:
            metabolism_invariants_failed_count += 1
        if "nightly_transaction_id" in policy_obs:
            tx_id = policy_obs.get("nightly_transaction_id")
            if not (isinstance(tx_id, str) and tx_id.strip()):
                transaction_missing_count += 1
        if "nightly_transaction_atomic" in policy_obs:
            tx_atomic = policy_obs.get("nightly_transaction_atomic")
            if isinstance(tx_atomic, bool) and not tx_atomic:
                transaction_non_atomic_count += 1
        if str(policy_obs.get("repair_status") or "").lower() == "applied" and "repair_plan_id" in policy_obs:
            repair_plan_id = policy_obs.get("repair_plan_id")
            if not (isinstance(repair_plan_id, str) and repair_plan_id.strip()):
                repair_plan_missing_count += 1
            if "repair_ops_digest" in policy_obs:
                repair_ops_digest = policy_obs.get("repair_ops_digest")
                if not (isinstance(repair_ops_digest, str) and repair_ops_digest.strip()):
                    repair_ops_digest_missing_count += 1

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
            "metabolism_invariants_failed_count": 0,
            "transaction_missing_count": 0,
            "transaction_non_atomic_count": 0,
            "repair_plan_missing_count": 0,
            "repair_ops_digest_missing_count": 0,
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
            and metabolism_invariants_failed_count == 0
            and transaction_missing_count == 0
            and transaction_non_atomic_count == 0
            and repair_plan_missing_count == 0
            and repair_ops_digest_missing_count == 0
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
        "metabolism_invariants_failed_count": metabolism_invariants_failed_count,
        "transaction_missing_count": transaction_missing_count,
        "transaction_non_atomic_count": transaction_non_atomic_count,
        "repair_plan_missing_count": repair_plan_missing_count,
        "repair_ops_digest_missing_count": repair_ops_digest_missing_count,
        "phase_transition_fp_stale_count": phase_transition_fp_stale_count,
        "phase_override_applied_count": phase_override_applied_count,
        "warnings": (
            ["PHASE_OVERRIDE_APPLIED"]
            if phase_override_applied_count > 0
            else []
        ),
    }


def _immune_guard_summary(records: List[Dict[str, Any]], day_key: str) -> Dict[str, Any]:
    total = 0
    quarantine_pruned_sum = 0
    immune_guard_pruned_sum = 0
    repeat_hit_rate_values: list[float] = []
    for row in records:
        policy_obs = (((row.get("policy") or {}).get("observations") or {}).get("hub") or {})
        if not isinstance(policy_obs, dict):
            continue
        if str(policy_obs.get("day_key") or "") != day_key:
            continue
        if str(policy_obs.get("operation") or "") != "run_nightly":
            continue
        total += 1
        quarantine_pruned_sum += int(policy_obs.get("quarantine_pruned_count") or 0)
        immune_guard_pruned_sum += int(policy_obs.get("immune_guard_pruned_count") or 0)
        value = policy_obs.get("repeat_hit_rate")
        if isinstance(value, (int, float)):
            repeat_hit_rate_values.append(float(value))
    avg_repeat = (sum(repeat_hit_rate_values) / float(len(repeat_hit_rate_values))) if repeat_hit_rate_values else 0.0
    return {
        "events_checked": int(total),
        "quarantine_pruned_count": int(quarantine_pruned_sum),
        "immune_guard_pruned_count": int(immune_guard_pruned_sum),
        "repeat_hit_rate": float(round(avg_repeat, 6)),
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
    lazy_rag_sat_ratios: List[float] = []
    uncertainty_reason_counts: Counter[str] = Counter()
    confidence_values: List[float] = []
    confidence_label_counts: Counter[str] = Counter()
    try:
        conf_low_max = float(os.getenv("EQNET_UNCERTAINTY_CONFIDENCE_LOW_MAX", "0.54"))
    except (TypeError, ValueError):
        conf_low_max = 0.54
    try:
        conf_mid_max = float(os.getenv("EQNET_UNCERTAINTY_CONFIDENCE_MID_MAX", "0.79"))
    except (TypeError, ValueError):
        conf_mid_max = 0.79
    if not (0.0 <= conf_low_max < conf_mid_max <= 1.0):
        conf_low_max = 0.54
        conf_mid_max = 0.79

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
            sat_ratio = _extract_lazy_rag_sat_ratio(row)
            if sat_ratio is not None:
                lazy_rag_sat_ratios.append(sat_ratio)
            reasons = _extract_uncertainty_reasons(row)
            if reasons:
                uncertainty_reason_counts.update(reasons)
            confidence_val = _extract_confidence_value(row)
            if confidence_val is not None:
                confidence_values.append(confidence_val)
                confidence_label_counts.update(
                    [_confidence_label(confidence_val, low_max=conf_low_max, mid_max=conf_mid_max)]
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

    mecpe_audit = _audit_mecpe(cfg)

    ru_v0_summary = _ru_v0_summary(records)

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
    immune_guard = _immune_guard_summary(records, cfg.date_yyyy_mm_dd)
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
    if int(memory_thermo_contract.get("metabolism_invariants_failed_count") or 0) > 0:
        health = _bump_health(
            health,
            level="YELLOW",
            reason="metabolism invariants violated",
        )
    if int(memory_thermo_contract.get("transaction_missing_count") or 0) > 0:
        health = _bump_health(
            health,
            level="YELLOW",
            reason="nightly transaction id missing",
        )
    if int(memory_thermo_contract.get("transaction_non_atomic_count") or 0) > 0:
        health = _bump_health(
            health,
            level="YELLOW",
            reason="nightly transaction marked non-atomic",
        )
    if int(memory_thermo_contract.get("repair_plan_missing_count") or 0) > 0:
        health = _bump_health(
            health,
            level="YELLOW",
            reason="repair plan id missing for applied repair",
        )
    if int(memory_thermo_contract.get("repair_ops_digest_missing_count") or 0) > 0:
        health = _bump_health(
            health,
            level="YELLOW",
            reason="repair ops digest missing for applied repair",
        )
    if int(memory_thermo_contract.get("phase_transition_fp_stale_count") or 0) > 0:
        health = _bump_health(
            health,
            level="YELLOW",
            reason="phase transition without projection fingerprint update detected",
        )
    tool_attempt_count = int(ru_v0_summary.get("tool_call_attempt_count") or 0)
    tool_blocked_count = int(ru_v0_summary.get("tool_call_blocked_count") or 0)
    if tool_attempt_count > 0:
        block_rate = tool_blocked_count / float(tool_attempt_count)
        if block_rate > float(thresholds.get("online_delta_block_rate_yellow", 0.8)):
            health = _bump_health(
                health,
                level="YELLOW",
                reason=f"online delta tool block rate high ({block_rate:.2f})",
            )
    forced_human_confirm_count = int(ru_v0_summary.get("forced_human_confirm_count") or 0)
    if forced_human_confirm_count > int(thresholds.get("forced_human_confirm_count_yellow", 20)):
        health = _bump_health(
            health,
            level="YELLOW",
            reason=f"forced_human_confirm count high ({forced_human_confirm_count})",
        )
    mecpe_contract_errors = (mecpe_audit.get("contract_errors") or {})
    if int(mecpe_contract_errors.get("total") or 0) > 0:
        top_type = str(mecpe_contract_errors.get("top_type") or "unknown")
        health = _bump_health(
            health,
            level="YELLOW",
            reason=f"mecpe contract errors detected (top={top_type})",
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
    promotion_policy = (
        load_promotion_policy(cfg.promotion_policy_path)
        if cfg.promotion_policy_path is not None
        else load_promotion_policy(Path("configs/online_delta_promotion_v0.yaml"))
    )
    promotion_summary = summarize_online_delta_effectiveness(records, policy=promotion_policy)
    promotion_decisions = decide_promotions(promotion_summary, policy=promotion_policy)
    if cfg.rule_delta_path is not None:
        rule_delta_path = cfg.rule_delta_path
    elif cfg.state_dir is not None:
        rule_delta_path = cfg.state_dir / "rule_delta.v0.jsonl"
    else:
        rule_delta_path = Path("state") / "rule_delta.v0.jsonl"
    promotion_append = append_rule_delta_promotions(
        rule_delta_path=rule_delta_path,
        decisions=promotion_decisions,
        day_key=cfg.date_yyyy_mm_dd,
        timestamp_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
    )
    promotion_payload = {
        "policy": promotion_policy,
        "baseline": promotion_summary.get("baseline") or {},
        "promotion_candidates_topk": promotion_summary.get("candidates") or [],
        "promotion_decisions": [item.to_dict() for item in promotion_decisions],
        "rule_delta_append_result": promotion_append,
    }
    lazy_rag_sat_ratio_p95 = _percentile(lazy_rag_sat_ratios, 0.95)
    confidence_total = len(confidence_values)
    low_count = int(confidence_label_counts.get("low", 0))
    mid_count = int(confidence_label_counts.get("mid", 0))
    high_count = int(confidence_label_counts.get("high", 0))

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
        "immune_guard": immune_guard,
        "separation": _separation_governance_audit(
            think_log_path=cfg.think_log_path,
            act_log_path=cfg.act_log_path,
        ),
        "ru_v0_summary": ru_v0_summary,
        "qualia_gate_stats": qualia_gate_stats,
        "lazy_rag_sat_ratio_p95": round(lazy_rag_sat_ratio_p95, 3),
        "lazy_rag_sat_ratio_count": len(lazy_rag_sat_ratios),
        "uncertainty_reason_top3": [
            {"reason": reason, "count": int(count)}
            for reason, count in uncertainty_reason_counts.most_common(3)
        ],
        "uncertainty_confidence": {
            "total": confidence_total,
            "low": low_count,
            "mid": mid_count,
            "high": high_count,
            "low_ratio": round((low_count / confidence_total), 3) if confidence_total else 0.0,
            "mid_ratio": round((mid_count / confidence_total), 3) if confidence_total else 0.0,
            "high_ratio": round((high_count / confidence_total), 3) if confidence_total else 0.0,
            "thresholds": {"low_max": conf_low_max, "mid_max": conf_mid_max},
        },
        "mecpe_audit": mecpe_audit,
        "online_delta_promotion": promotion_payload,
    }

    cfg.out_root.mkdir(parents=True, exist_ok=True)
    out_path = cfg.out_root / f"nightly_audit_{cfg.date_yyyy_mm_dd}.json"
    out_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


__all__ = ["NightlyAuditConfig", "generate_audit"]
