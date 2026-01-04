#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate a nightly audit report from activation traces and telemetry."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List
from collections import Counter

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.nightly_report import generate_recall_report
from runtime.config import load_runtime_cfg
from eqnet.telemetry.nightly_audit import NightlyAuditConfig, generate_audit


def _resolve_log_path(template: str, *, day: date) -> Path:
    if "%" not in template:
        return Path(template)
    return Path(day.strftime(template))

def _latest_trace_day(trace_root: Path) -> date | None:
    if not trace_root.exists():
        return None
    candidates: List[date] = []
    for child in trace_root.iterdir():
        if not child.is_dir():
            continue
        try:
            candidates.append(date.fromisoformat(child.name))
        except ValueError:
            continue
    return max(candidates) if candidates else None


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def _load_telemetry(paths: List[Path]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for path in paths:
        entries.extend(_read_jsonl(path))
    return entries


def _split_halves(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"first_mean": 0.0, "second_mean": 0.0, "delta": 0.0}
    mid = max(1, len(values) // 2)
    first = values[:mid]
    second = values[mid:] or first
    first_mean = sum(first) / len(first)
    second_mean = sum(second) / len(second)
    return {
        "first_mean": round(first_mean, 3),
        "second_mean": round(second_mean, 3),
        "delta": round(second_mean - first_mean, 3),
    }

def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * pct))
    return float(ordered[min(max(idx, 0), len(ordered) - 1)])


def _recovery_steps(drives: List[float], limit: float) -> Dict[str, float]:
    if not drives:
        return {"count": 0, "avg_steps": 0.0, "max_steps": 0.0}
    durations: List[int] = []
    active = False
    start = 0
    for idx, value in enumerate(drives):
        above = value > limit
        if above and not active:
            active = True
            start = idx
        elif active and not above:
            durations.append(idx - start)
            active = False
    if active:
        durations.append(len(drives) - start)
    if not durations:
        return {"count": 0, "avg_steps": 0.0, "max_steps": 0.0}
    avg_steps = sum(durations) / len(durations)
    return {
        "count": len(durations),
        "avg_steps": round(avg_steps, 2),
        "max_steps": int(max(durations)),
    }


def _kpi_summary(
    events: List[Dict[str, Any]],
    *,
    start_step: int | None = None,
    end_step: int | None = None,
) -> Dict[str, Any]:
    steps = []
    for e in events:
        if e.get("event") != "minimal_heartos.step":
            continue
        data = e.get("data", {}) if isinstance(e.get("data"), dict) else {}
        step_index = data.get("step_index")
        if isinstance(step_index, int):
            if start_step is not None and step_index < start_step:
                continue
            if end_step is not None and step_index >= end_step:
                continue
        steps.append(e)
    step_count = len(steps)
    cancel_count = 0
    cancel_causes: Dict[str, int] = {}
    drives: List[float] = []
    drive_limit = None
    for entry in steps:
        data = entry.get("data", {}) if isinstance(entry.get("data"), dict) else {}
        decision = data.get("decision")
        if decision == "cancel":
            cancel_count += 1
            cause = data.get("cancel_reason") or "unknown"
            cancel_causes[cause] = cancel_causes.get(cause, 0) + 1
        drive_val = data.get("drive")
        if isinstance(drive_val, (int, float)):
            drives.append(float(drive_val))
        if drive_limit is None and isinstance(data.get("drive_limit"), (int, float)):
            drive_limit = float(data.get("drive_limit"))
    drive_limit = drive_limit if drive_limit is not None else 0.75
    recovery = _recovery_steps(drives, drive_limit)
    drive_over = sum(1 for v in drives if v > drive_limit)
    cancel_rate = (cancel_count / step_count) if step_count else 0.0
    return {
        "step_count": step_count,
        "cancel_count": cancel_count,
        "cancel_rate": round(cancel_rate, 3),
        "cancel_causes": cancel_causes,
        "drive_limit": drive_limit,
        "drive_over_limit": drive_over,
        "drive_recovery": recovery,
        "drive_trend": _split_halves(drives),
    }

def _segment_key(step_index: int | None, transition_turn_index: int | None) -> str:
    if transition_turn_index is None or step_index is None:
        return "all"
    return "pre" if step_index < transition_turn_index else "post"


def _segment_telemetry(
    events: List[Dict[str, Any]], transition_turn_index: int | None
) -> Dict[str, List[Dict[str, Any]]]:
    segments: Dict[str, List[Dict[str, Any]]] = {}
    for event in events:
        if event.get("event") != "minimal_heartos.step":
            continue
        data = event.get("data", {}) if isinstance(event.get("data"), dict) else {}
        step_index = data.get("step_index")
        key = _segment_key(step_index if isinstance(step_index, int) else None, transition_turn_index)
        segments.setdefault(key, []).append(event)
    return segments


def _cancel_causes_stats(events: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    causes: Dict[str, int] = {}
    total = 0
    for event in events:
        data = event.get("data", {}) if isinstance(event.get("data"), dict) else {}
        if data.get("decision") != "cancel":
            continue
        total += 1
        cause = data.get("cancel_reason") or "unknown"
        causes[cause] = causes.get(cause, 0) + 1
    if total == 0:
        return {}
    return {
        k: {"count": v, "ratio": round(v / total, 3)}
        for k, v in causes.items()
    }

def _mean_field(events: List[Dict[str, Any]], field: str) -> float:
    values: List[float] = []
    for event in events:
        data = event.get("data", {}) if isinstance(event.get("data"), dict) else {}
        value = data.get(field)
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def _veto_streak_stats(events: List[Dict[str, Any]]) -> Dict[str, float]:
    streaks: List[int] = []
    current = 0
    accept_count = 0
    for event in events:
        data = event.get("data", {}) if isinstance(event.get("data"), dict) else {}
        if data.get("decision") == "cancel":
            current += 1
        else:
            if current:
                streaks.append(current)
                current = 0
            accept_count += 1
    if current:
        streaks.append(current)
    if not streaks:
        return {"max": 0, "avg": 0.0, "accept_count": accept_count}
    return {
        "max": int(max(streaks)),
        "avg": round(sum(streaks) / len(streaks), 2),
        "accept_count": accept_count,
    }


def _executed_boundary_stats(decision_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    boundaries: List[float] = []
    entries: List[Dict[str, Any]] = []
    for record in decision_records:
        prospection = record.get("prospection") or {}
        if not bool(prospection.get("accepted")):
            continue
        boundary = record.get("boundary") or {}
        score = boundary.get("score")
        if not isinstance(score, (int, float)):
            continue
        reasons = boundary.get("reasons") or {}
        entry = {
            "turn_id": record.get("turn_id"),
            "world_type": record.get("world_type"),
            "boundary_score": float(score),
            "hazard_score": reasons.get("hazard_score"),
            "drive": reasons.get("drive"),
            "drive_norm": reasons.get("drive_norm"),
            "risk": reasons.get("risk"),
            "uncertainty": reasons.get("uncertainty"),
        }
        entries.append(entry)
        boundaries.append(float(score))
    if not boundaries:
        return {"executed_count": 0}
    ordered = sorted(boundaries)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        median = ordered[mid]
    else:
        median = 0.5 * (ordered[mid - 1] + ordered[mid])
    max_entry = max(entries, key=lambda item: item.get("boundary_score", 0.0))
    return {
        "executed_count": len(boundaries),
        "executed_boundary_min": round(ordered[0], 3),
        "executed_boundary_p50": round(median, 3),
        "executed_boundary_max": round(ordered[-1], 3),
        "executed_boundary_max_example": max_entry,
    }


def _executed_boundary_by_world(
    decision_records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    by_world: Dict[str, List[float]] = {}
    for record in decision_records:
        prospection = record.get("prospection") or {}
        if not bool(prospection.get("accepted")):
            continue
        world_type = record.get("world_type") or "unknown"
        boundary = record.get("boundary") or {}
        score = boundary.get("score")
        if not isinstance(score, (int, float)):
            continue
        by_world.setdefault(world_type, []).append(float(score))
    summary: Dict[str, Dict[str, float]] = {}
    for world_type, scores in by_world.items():
        if not scores:
            continue
        summary[world_type] = {
            "executed_count": len(scores),
            "executed_boundary_max": round(max(scores), 3),
        }
    return summary


def _decision_boundary_stats(decision_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_scores: List[float] = []
    cancel_scores: List[float] = []
    for record in decision_records:
        boundary = record.get("boundary") or {}
        score = boundary.get("score")
        if not isinstance(score, (int, float)):
            continue
        all_scores.append(float(score))
        prospection = record.get("prospection") or {}
        if not bool(prospection.get("accepted")):
            cancel_scores.append(float(score))
    return {
        "all_decision_boundary_max": round(max(all_scores), 3) if all_scores else 0.0,
        "cancel_boundary_max": round(max(cancel_scores), 3) if cancel_scores else 0.0,
    }


def _decision_boundary_by_world(
    decision_records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    all_scores: Dict[str, List[float]] = {}
    cancel_scores: Dict[str, List[float]] = {}
    for record in decision_records:
        boundary = record.get("boundary") or {}
        score = boundary.get("score")
        if not isinstance(score, (int, float)):
            continue
        world_type = record.get("world_type") or "unknown"
        all_scores.setdefault(world_type, []).append(float(score))
        prospection = record.get("prospection") or {}
        if not bool(prospection.get("accepted")):
            cancel_scores.setdefault(world_type, []).append(float(score))
    payload: Dict[str, Dict[str, float]] = {}
    worlds = set(all_scores.keys()) | set(cancel_scores.keys())
    for world_type in worlds:
        payload[world_type] = {
            "all_decision_boundary_max": round(max(all_scores.get(world_type, [0.0])), 3),
            "cancel_boundary_max": round(max(cancel_scores.get(world_type, [0.0])), 3),
        }
    return payload


def _decision_score_stats(decision_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    score_all: List[float] = []
    score_exec: List[float] = []
    score_cancel: List[float] = []
    u_hat_all: List[float] = []
    u_hat_exec: List[float] = []
    u_hat_cancel: List[float] = []
    veto_all: List[float] = []
    veto_exec: List[float] = []
    veto_cancel: List[float] = []
    for record in decision_records:
        decision = record.get("decision") or {}
        score = decision.get("score")
        u_hat = decision.get("u_hat")
        veto = decision.get("veto_score")
        if isinstance(score, (int, float)):
            score_all.append(float(score))
        if isinstance(u_hat, (int, float)):
            u_hat_all.append(float(u_hat))
        if isinstance(veto, (int, float)):
            veto_all.append(float(veto))
        prospection = record.get("prospection") or {}
        accepted = bool(prospection.get("accepted"))
        if accepted:
            if isinstance(score, (int, float)):
                score_exec.append(float(score))
            if isinstance(u_hat, (int, float)):
                u_hat_exec.append(float(u_hat))
            if isinstance(veto, (int, float)):
                veto_exec.append(float(veto))
        else:
            if isinstance(score, (int, float)):
                score_cancel.append(float(score))
            if isinstance(u_hat, (int, float)):
                u_hat_cancel.append(float(u_hat))
            if isinstance(veto, (int, float)):
                veto_cancel.append(float(veto))
    return {
        "decision_score_executed_max": round(max(score_exec), 3) if score_exec else 0.0,
        "decision_score_cancel_max": round(max(score_cancel), 3) if score_cancel else 0.0,
        "decision_score_all_max": round(max(score_all), 3) if score_all else 0.0,
        "u_hat_executed_max": round(max(u_hat_exec), 3) if u_hat_exec else 0.0,
        "u_hat_cancel_max": round(max(u_hat_cancel), 3) if u_hat_cancel else 0.0,
        "u_hat_all_max": round(max(u_hat_all), 3) if u_hat_all else 0.0,
        "veto_score_executed_min": round(min(veto_exec), 3) if veto_exec else 0.0,
        "veto_score_cancel_max": round(max(veto_cancel), 3) if veto_cancel else 0.0,
        "veto_score_all_max": round(max(veto_all), 3) if veto_all else 0.0,
    }


def _decision_score_by_world(
    decision_records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    score_exec: Dict[str, List[float]] = {}
    score_cancel: Dict[str, List[float]] = {}
    u_hat_exec: Dict[str, List[float]] = {}
    u_hat_cancel: Dict[str, List[float]] = {}
    veto_exec: Dict[str, List[float]] = {}
    veto_cancel: Dict[str, List[float]] = {}
    for record in decision_records:
        decision = record.get("decision") or {}
        score = decision.get("score")
        u_hat = decision.get("u_hat")
        veto = decision.get("veto_score")
        world_type = record.get("world_type") or "unknown"
        prospection = record.get("prospection") or {}
        accepted = bool(prospection.get("accepted"))
        if accepted:
            if isinstance(score, (int, float)):
                score_exec.setdefault(world_type, []).append(float(score))
            if isinstance(u_hat, (int, float)):
                u_hat_exec.setdefault(world_type, []).append(float(u_hat))
            if isinstance(veto, (int, float)):
                veto_exec.setdefault(world_type, []).append(float(veto))
        else:
            if isinstance(score, (int, float)):
                score_cancel.setdefault(world_type, []).append(float(score))
            if isinstance(u_hat, (int, float)):
                u_hat_cancel.setdefault(world_type, []).append(float(u_hat))
            if isinstance(veto, (int, float)):
                veto_cancel.setdefault(world_type, []).append(float(veto))
    payload: Dict[str, Dict[str, float]] = {}
    worlds = set(score_exec.keys()) | set(score_cancel.keys()) | set(u_hat_exec.keys()) | set(u_hat_cancel.keys())
    for world_type in worlds:
        payload[world_type] = {
            "decision_score_executed_max": round(max(score_exec.get(world_type, [0.0])), 3),
            "decision_score_cancel_max": round(max(score_cancel.get(world_type, [0.0])), 3),
            "u_hat_executed_max": round(max(u_hat_exec.get(world_type, [0.0])), 3),
            "u_hat_cancel_max": round(max(u_hat_cancel.get(world_type, [0.0])), 3),
            "veto_score_executed_min": round(min(veto_exec.get(world_type, [0.0])), 3),
            "veto_score_cancel_max": round(max(veto_cancel.get(world_type, [0.0])), 3),
        }
    return payload


def _ru_v0_summary(trace_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    gate_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    missing_events = 0
    ru_events = 0
    for record in trace_records:
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


def _summary_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "p50": 0.0, "max": 0.0}
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        median = ordered[mid]
    else:
        median = 0.5 * (ordered[mid - 1] + ordered[mid])
    return {
        "min": round(ordered[0], 3),
        "p50": round(median, 3),
        "max": round(ordered[-1], 3),
    }


def _high_boundary_cancel_stats(
    decision_records: List[Dict[str, Any]],
    *,
    threshold: float = 0.55,
) -> Dict[str, Any]:
    score_values: List[float] = []
    u_hat_values: List[float] = []
    veto_values: List[float] = []
    risk_values: List[float] = []
    uncert_values: List[float] = []
    count = 0
    for record in decision_records:
        prospection = record.get("prospection") or {}
        if bool(prospection.get("accepted")):
            continue
        boundary = record.get("boundary") or {}
        boundary_score = boundary.get("score")
        if not isinstance(boundary_score, (int, float)):
            continue
        if float(boundary_score) < threshold:
            continue
        decision = record.get("decision") or {}
        score = decision.get("score")
        u_hat = decision.get("u_hat")
        veto = decision.get("veto_score")
        reasons = boundary.get("reasons") or {}
        risk = reasons.get("risk")
        uncertainty = reasons.get("uncertainty")
        count += 1
        if isinstance(score, (int, float)):
            score_values.append(float(score))
        if isinstance(u_hat, (int, float)):
            u_hat_values.append(float(u_hat))
        if isinstance(veto, (int, float)):
            veto_values.append(float(veto))
        if isinstance(risk, (int, float)):
            risk_values.append(float(risk))
        if isinstance(uncertainty, (int, float)):
            uncert_values.append(float(uncertainty))
    return {
        "threshold": round(float(threshold), 3),
        "count": count,
        "decision_score": _summary_stats(score_values),
        "u_hat": _summary_stats(u_hat_values),
        "veto_score": _summary_stats(veto_values),
        "risk": _summary_stats(risk_values),
        "uncertainty": _summary_stats(uncert_values),
    }


def _high_boundary_cancel_by_world(
    decision_records: List[Dict[str, Any]],
    *,
    threshold: float = 0.55,
) -> Dict[str, Dict[str, Any]]:
    payload: Dict[str, Dict[str, Any]] = {}
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in decision_records:
        prospection = record.get("prospection") or {}
        if bool(prospection.get("accepted")):
            continue
        boundary = record.get("boundary") or {}
        boundary_score = boundary.get("score")
        if not isinstance(boundary_score, (int, float)):
            continue
        if float(boundary_score) < threshold:
            continue
        world_type = record.get("world_type") or "unknown"
        grouped.setdefault(world_type, []).append(record)
    for world_type, records in grouped.items():
        payload[world_type] = _high_boundary_cancel_stats(records, threshold=threshold)
    return payload


def _cancel_reason_summary(
    decision_records: List[Dict[str, Any]],
    *,
    max_examples: int = 3,
) -> Dict[str, Any]:
    reasons: Dict[str, int] = {}
    examples: Dict[str, List[Dict[str, Any]]] = {}
    total = 0
    for record in decision_records:
        prospection = record.get("prospection") or {}
        if bool(prospection.get("accepted")):
            continue
        total += 1
        boundary = record.get("boundary") or {}
        reason_map = boundary.get("reasons") or {}
        reason_key = "unknown"
        reason_value = None
        if isinstance(reason_map, dict) and reason_map:
            best = max(
                reason_map.items(),
                key=lambda item: item[1] if isinstance(item[1], (int, float)) else float("-inf"),
            )
            if isinstance(best[1], (int, float)):
                reason_key = str(best[0])
                reason_value = float(best[1])
        reasons[reason_key] = reasons.get(reason_key, 0) + 1
        if len(examples.get(reason_key, [])) < max_examples:
            examples.setdefault(reason_key, []).append(
                {
                    "turn_id": record.get("turn_id"),
                    "world_type": record.get("world_type"),
                    "boundary_score": boundary.get("score"),
                    "reason_value": reason_value,
                    "reasons": reason_map if isinstance(reason_map, dict) else {},
                }
            )
    return {
        "total": total,
        "reasons": reasons,
        "max_reason_examples": examples,
    }


def _cancel_reason_by_world(
    decision_records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    by_world: Dict[str, Dict[str, int]] = {}
    totals: Dict[str, int] = {}
    for record in decision_records:
        prospection = record.get("prospection") or {}
        if bool(prospection.get("accepted")):
            continue
        world_type = record.get("world_type") or "unknown"
        totals[world_type] = totals.get(world_type, 0) + 1
        boundary = record.get("boundary") or {}
        reason_map = boundary.get("reasons") or {}
        reason_key = "unknown"
        if isinstance(reason_map, dict) and reason_map:
            best = max(
                reason_map.items(),
                key=lambda item: item[1] if isinstance(item[1], (int, float)) else float("-inf"),
            )
            if isinstance(best[1], (int, float)):
                reason_key = str(best[0])
        by_world.setdefault(world_type, {})
        by_world[world_type][reason_key] = by_world[world_type].get(reason_key, 0) + 1
    payload: Dict[str, Dict[str, Any]] = {}
    for world_type, reasons in by_world.items():
        payload[world_type] = {
            "total": totals.get(world_type, 0),
            "reasons": reasons,
        }
    return payload


def _load_trace_v1_events(trace_root: Path, day: date) -> List[Dict[str, Any]]:
    day_dir = trace_root / day.isoformat()
    if not day_dir.exists():
        return []
    records: List[Dict[str, Any]] = []
    for path in sorted(day_dir.glob("*.jsonl")):
        for record in _read_jsonl(path):
            if "trace_file" not in record:
                record["trace_file"] = str(path)
            records.append(record)
    return records


def _transition_events(trace_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for record in trace_records:
        if record.get("event_type") == "world_transition" or record.get("source_loop") == "world_transition":
            events.append(record)
    return events


def _decision_cycle_records(
    trace_records: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    return [
        record
        for record in trace_records
        if record.get("event_type") == "decision_cycle"
    ]


def _deviant_records(trace_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        record
        for record in trace_records
        if record.get("event_type") == "deviant_event"
    ]


def _classify_deviant(record: Dict[str, Any]) -> str:
    deviant = record.get("deviant") or {}
    mode = None
    if isinstance(deviant, dict):
        mode = deviant.get("mode")
    if isinstance(mode, str):
        if mode == "boundary_only":
            return "boundary"
        if mode == "risk_only":
            return "risk"
    reasons = deviant.get("reasons") if isinstance(deviant, dict) else {}
    if isinstance(reasons, dict):
        has_boundary = "boundary_score" in reasons
        has_risk = "risk" in reasons or "uncertainty" in reasons
        if has_boundary and not has_risk:
            return "boundary"
        if has_risk and not has_boundary:
            return "risk"
        if has_boundary and has_risk:
            return "mixed"
    return "unknown"


def _recent_days(trace_root: Path, end_day: date, days: int) -> List[date]:
    if days <= 0:
        return []
    selected: List[date] = []
    for offset in range(days):
        candidate = end_day.fromordinal(end_day.toordinal() - offset)
        if (trace_root / candidate.isoformat()).exists():
            selected.append(candidate)
    return sorted(selected)


def _load_trace_v1_for_days(trace_root: Path, days: List[date]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for day in days:
        records.extend(_load_trace_v1_events(trace_root, day))
    return records


def _deviant_summary(
    deviant_records: List[Dict[str, Any]],
    decision_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    by_world: Dict[str, int] = {}
    reason_counts: Dict[str, int] = {}
    for record in deviant_records:
        world_type = record.get("world_type") or "unknown"
        by_world[world_type] = by_world.get(world_type, 0) + 1
        deviant = record.get("deviant") or {}
        reasons = deviant.get("reasons") if isinstance(deviant, dict) else {}
        if isinstance(reasons, dict):
            for key in reasons.keys():
                reason_counts[key] = reason_counts.get(key, 0) + 1
    decision_total = len(decision_records)
    decision_executed = sum(
        1 for record in decision_records if bool((record.get("prospection") or {}).get("accepted"))
    )
    deviant_count = len(deviant_records)
    ratio = (deviant_count / decision_executed) if decision_executed else 0.0
    return {
        "count": deviant_count,
        "ratio": round(ratio, 3),
        "decision_cycle_total": decision_total,
        "decision_cycle_executed": decision_executed,
        "by_world": by_world,
        "reasons": reason_counts,
    }


def _deviant_window_stats(
    trace_records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    decision_by_world: Dict[str, int] = {}
    decision_exec_by_world: Dict[str, int] = {}
    deviant_by_world: Dict[str, int] = {}
    for record in trace_records:
        world_type = record.get("world_type") or "unknown"
        if record.get("event_type") == "decision_cycle":
            decision_by_world[world_type] = decision_by_world.get(world_type, 0) + 1
            if bool((record.get("prospection") or {}).get("accepted")):
                decision_exec_by_world[world_type] = decision_exec_by_world.get(world_type, 0) + 1
        elif record.get("event_type") == "deviant_event":
            deviant_by_world[world_type] = deviant_by_world.get(world_type, 0) + 1
    payload: Dict[str, Dict[str, Any]] = {}
    all_worlds = set(deviant_by_world.keys()) | set(decision_by_world.keys())
    for world_type in sorted(all_worlds):
        deviant_count = deviant_by_world.get(world_type, 0)
        decision_count = decision_by_world.get(world_type, 0)
        decision_executed = decision_exec_by_world.get(world_type, 0)
        ratio = (deviant_count / decision_executed) if decision_executed else 0.0
        payload[world_type] = {
            "deviant_count": deviant_count,
            "decision_count": decision_count,
            "decision_executed": decision_executed,
            "ratio": round(ratio, 3),
        }
    return payload


def _deviant_window_stats_for_records(
    deviant_records: List[Dict[str, Any]],
    decision_records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    deviant_by_world: Dict[str, int] = {}
    decision_by_world: Dict[str, int] = {}
    decision_exec_by_world: Dict[str, int] = {}
    for record in decision_records:
        world_type = record.get("world_type") or "unknown"
        decision_by_world[world_type] = decision_by_world.get(world_type, 0) + 1
        if bool((record.get("prospection") or {}).get("accepted")):
            decision_exec_by_world[world_type] = decision_exec_by_world.get(world_type, 0) + 1
    for record in deviant_records:
        world_type = record.get("world_type") or "unknown"
        deviant_by_world[world_type] = deviant_by_world.get(world_type, 0) + 1
    payload: Dict[str, Dict[str, Any]] = {}
    all_worlds = set(deviant_by_world.keys()) | set(decision_by_world.keys())
    for world_type in sorted(all_worlds):
        deviant_count = deviant_by_world.get(world_type, 0)
        decision_count = decision_by_world.get(world_type, 0)
        decision_executed = decision_exec_by_world.get(world_type, 0)
        ratio = (deviant_count / decision_executed) if decision_executed else 0.0
        payload[world_type] = {
            "deviant_count": deviant_count,
            "decision_count": decision_count,
            "decision_executed": decision_executed,
            "ratio": round(ratio, 3),
        }
    return payload


def _accept_contexts(
    trace_records: List[Dict[str, Any]],
    transition_turn_index: int | None,
    limit: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    contexts: Dict[str, List[Dict[str, Any]]] = {"pre": [], "post": [], "all": []}
    for record in trace_records:
        prospection = record.get("prospection") or {}
        if not bool(prospection.get("accepted")):
            continue
        turn_id = record.get("turn_id", "")
        step_index = None
        if isinstance(turn_id, str) and "-" in turn_id:
            try:
                step_index = int(turn_id.split("-")[-1])
            except ValueError:
                step_index = None
        key = _segment_key(step_index, transition_turn_index)
        entry = {
            "turn_id": turn_id,
            "scenario_id": record.get("scenario_id"),
            "world_type": record.get("world_type"),
            "boundary_score": (record.get("boundary") or {}).get("score"),
            "boundary_reasons": (record.get("boundary") or {}).get("reasons") or {},
            "trace_file": record.get("trace_file"),
        }
        contexts.setdefault(key, []).append(entry)
        contexts["all"].append(entry)
    for key in list(contexts.keys()):
        contexts[key] = contexts[key][:limit]
    return contexts


def _write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    kpi = payload.get("kpi", {})
    recall = payload.get("recall_report", {})
    nightly = payload.get("nightly_audit") or {}
    lines.append(f"# Nightly Audit ({payload.get('day')})")
    lines.append("")
    lines.append("## KPI")
    lines.append(f"- steps: {kpi.get('step_count', 0)}")
    lines.append(f"- cancel_rate: {kpi.get('cancel_rate', 0.0)}")
    lines.append(f"- drive_over_limit: {kpi.get('drive_over_limit', 0)}")
    lines.append(f"- drive_recovery_count: {kpi.get('drive_recovery', {}).get('count', 0)}")
    lines.append("")
    lines.append("## Cancel Causes")
    cancel_causes = kpi.get("cancel_causes", {})
    if cancel_causes:
        for key, value in sorted(cancel_causes.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Trace Coverage")
    lines.append(f"- trace_count: {payload.get('trace_count', 0)}")
    lines.append(f"- coverage_ratio: {payload.get('trace_coverage', 0.0)}")
    lines.append("")
    transitions = payload.get("transitions", {})
    lines.append("## World Transitions")
    lines.append(f"- transition_count: {transitions.get('count', 0)}")
    if transitions.get("first_ts") is not None:
        lines.append(f"- first_transition_ts: {transitions.get('first_ts')}")
    if transitions.get("transition_turn_index") is not None:
        lines.append(f"- transition_turn_index: {transitions.get('transition_turn_index')}")
    pre = transitions.get("pre_kpi")
    post = transitions.get("post_kpi")
    if isinstance(pre, dict) and isinstance(post, dict):
        lines.append(f"- pre_cancel_rate: {pre.get('cancel_rate', 0.0)}")
        lines.append(f"- post_cancel_rate: {post.get('cancel_rate', 0.0)}")
    lines.append("")
    deviant = payload.get("deviant", {})
    lines.append("## Deviant Events")
    lines.append(f"- count: {deviant.get('count', 0)}")
    lines.append(f"- ratio: {deviant.get('ratio', 0.0)}")
    lines.append(f"- decision_cycle_total: {deviant.get('decision_cycle_total', 0)}")
    lines.append(f"- decision_cycle_executed: {deviant.get('decision_cycle_executed', 0)}")
    by_world = deviant.get("by_world", {})
    if by_world:
        lines.append(f"- by_world: {by_world}")
    reasons = deviant.get("reasons", {})
    if reasons:
        lines.append(f"- reasons: {reasons}")
    lines.append("")
    resonance = payload.get("resonance", {})
    lines.append("## Resonance Events (risk)")
    lines.append(f"- count: {resonance.get('count', 0)}")
    lines.append(f"- ratio: {resonance.get('ratio', 0.0)}")
    lines.append(f"- decision_cycle_total: {resonance.get('decision_cycle_total', 0)}")
    lines.append(f"- decision_cycle_executed: {resonance.get('decision_cycle_executed', 0)}")
    res_by_world = resonance.get("by_world", {})
    if res_by_world:
        lines.append(f"- by_world: {res_by_world}")
    res_reasons = resonance.get("reasons", {})
    if res_reasons:
        lines.append(f"- reasons: {res_reasons}")
    lines.append("")
    executed = payload.get("executed_boundary", {})
    lines.append("## Executed Boundary Summary")
    if executed.get("executed_count", 0):
        lines.append(f"- executed_count: {executed.get('executed_count', 0)}")
        lines.append(f"- executed_boundary_min: {executed.get('executed_boundary_min', 0.0)}")
        lines.append(f"- executed_boundary_p50: {executed.get('executed_boundary_p50', 0.0)}")
        lines.append(f"- executed_boundary_max: {executed.get('executed_boundary_max', 0.0)}")
        example = executed.get("executed_boundary_max_example", {})
        if example:
            lines.append(
                "- executed_boundary_max_example: "
                f"{example}"
            )
    else:
        lines.append("- executed_count: 0")
    lines.append("")
    decision_bounds = payload.get("decision_boundary", {})
    lines.append("## Decision Boundary Max")
    lines.append(f"- all_decision_boundary_max: {decision_bounds.get('all_decision_boundary_max', 0.0)}")
    lines.append(f"- cancel_boundary_max: {decision_bounds.get('cancel_boundary_max', 0.0)}")
    lines.append("")
    decision_scores = payload.get("decision_score", {})
    lines.append("## Decision Score Summary")
    lines.append(f"- decision_score_executed_max: {decision_scores.get('decision_score_executed_max', 0.0)}")
    lines.append(f"- decision_score_cancel_max: {decision_scores.get('decision_score_cancel_max', 0.0)}")
    lines.append(f"- decision_score_all_max: {decision_scores.get('decision_score_all_max', 0.0)}")
    lines.append(f"- u_hat_executed_max: {decision_scores.get('u_hat_executed_max', 0.0)}")
    lines.append(f"- u_hat_cancel_max: {decision_scores.get('u_hat_cancel_max', 0.0)}")
    lines.append(f"- veto_score_executed_min: {decision_scores.get('veto_score_executed_min', 0.0)}")
    lines.append(f"- veto_score_cancel_max: {decision_scores.get('veto_score_cancel_max', 0.0)}")
    lines.append("")
    gate_stats = payload.get("qualia_gate_stats", {})
    if gate_stats:
        term = gate_stats.get("boundary_term", {})
        if term:
            lines.append("## Qualia Gate (Boundary)")
            lines.append(f"- boundary_term_mean: {term.get('mean', 0.0)}")
            lines.append(f"- boundary_term_p95: {term.get('p95', 0.0)}")
            lines.append(f"- boundary_term_max: {term.get('max', 0.0)}")
        ignored = gate_stats.get("boundary_curve_ignored", {})
        if ignored:
            lines.append(f"- boundary_curve_ignored_ratio: {ignored.get('ratio', 0.0)}")
        presence = gate_stats.get("presence", {})
        if presence:
            lines.append(f"- ack_silence_ratio: {presence.get('ack_silence_ratio', 0.0)}")
        lines.append("")
    high_boundary = payload.get("high_boundary_cancel", {})
    lines.append("## High Boundary Cancels (decision_cycle)")
    lines.append(f"- threshold: {high_boundary.get('threshold', 0.55)}")
    lines.append(f"- count: {high_boundary.get('count', 0)}")
    lines.append(f"- decision_score: {high_boundary.get('decision_score', {})}")
    lines.append(f"- u_hat: {high_boundary.get('u_hat', {})}")
    lines.append(f"- veto_score: {high_boundary.get('veto_score', {})}")
    lines.append(f"- risk: {high_boundary.get('risk', {})}")
    lines.append(f"- uncertainty: {high_boundary.get('uncertainty', {})}")
    lines.append("")
    cancel = payload.get("cancel_summary", {})
    lines.append("## Cancel Summary (trace_v1)")
    lines.append(f"- cancel_total: {cancel.get('total', 0)}")
    cancel_reasons = cancel.get("reasons", {})
    if cancel_reasons:
        lines.append(f"- cancel_reasons: {cancel_reasons}")
    else:
        lines.append("- cancel_reasons: {}")
    cancel_examples = cancel.get("max_reason_examples", {})
    if cancel_examples:
        lines.append(f"- cancel_reason_examples: {cancel_examples}")
    lines.append("")
    ru_v0 = payload.get("ru_v0_summary", {})
    if ru_v0:
        lines.append("## RU v0 Summary")
        lines.append(f"- ru_v0_events: {ru_v0.get('ru_v0_events', 0)}")
        lines.append(f"- gate_action_counts: {ru_v0.get('gate_action_counts', {})}")
        lines.append(f"- policy_version_counts: {ru_v0.get('policy_version_counts', {})}")
        lines.append(
            f"- missing_required_fields_events: {ru_v0.get('missing_required_fields_events', 0)}"
        )
        lines.append("")
    world_breakdown = payload.get("world_breakdown", {})
    lines.append("## World Breakdown")
    if world_breakdown:
        for world_type, summary in world_breakdown.items():
            lines.append(
                f"- {world_type}: executed_count={summary.get('executed_count', 0)}, "
                f"executed_boundary_max={summary.get('executed_boundary_max', 0.0)}, "
                f"all_decision_boundary_max={summary.get('all_decision_boundary_max', 0.0)}, "
                f"cancel_boundary_max={summary.get('cancel_boundary_max', 0.0)}, "
                f"decision_score_executed_max={summary.get('decision_score_executed_max', 0.0)}, "
                f"decision_score_cancel_max={summary.get('decision_score_cancel_max', 0.0)}, "
                f"u_hat_cancel_max={summary.get('u_hat_cancel_max', 0.0)}, "
                f"veto_score_cancel_max={summary.get('veto_score_cancel_max', 0.0)}, "
                f"deviant_count={summary.get('deviant_count', 0)}, "
                f"cancel_total={summary.get('cancel_total', 0)}, "
                f"cancel_reasons={summary.get('cancel_reasons', {})}"
            )
    else:
        lines.append("- none")
    lines.append("")
    proposals = payload.get("world_prior_proposals", [])
    lines.append("## World Prior Proposals (manual)")
    if proposals:
        for proposal in proposals:
            world_type = proposal.get("world_type", "unknown")
            ratio = proposal.get("ratio", 0.0)
            deviant_count = proposal.get("deviant_count", 0)
            decision_count = proposal.get("decision_count", 0)
            lines.append(
                f"- {world_type}: deviant={deviant_count}, decisions={decision_count}, ratio={ratio}"
            )
    else:
        lines.append("- none")
    lines.append("")
    resonance_notices = payload.get("resonance_notices", [])
    lines.append("## Resonance Notices (manual)")
    if resonance_notices:
        for notice in resonance_notices:
            world_type = notice.get("world_type", "unknown")
            ratio = notice.get("ratio", 0.0)
            deviant_count = notice.get("deviant_count", 0)
            decision_count = notice.get("decision_count", 0)
            lines.append(
                f"- {world_type}: resonance={deviant_count}, decisions={decision_count}, ratio={ratio}"
            )
    else:
        lines.append("- none")
    lines.append("")
    segments = payload.get("segments", {})
    if segments:
        lines.append("## Segment Summary")
        for key in ("pre", "post", "all"):
            if key not in segments:
                continue
            seg = segments[key]
            lines.append(f"- {key}: steps={seg.get('steps', 0)}, cancel_rate={seg.get('cancel_rate', 0.0)}")
        lines.append("")
        lines.append("## Segment Means")
        for key in ("pre", "post", "all"):
            if key not in segments:
                continue
            seg = segments[key]
            lines.append(
                f"- {key}: mean_drive={seg.get('mean_drive', 0.0)}, "
                f"mean_uncertainty={seg.get('mean_uncertainty', 0.0)}"
            )
        lines.append("")
        lines.append("## Cancel Causes Ratio")
        for key in ("pre", "post", "all"):
            if key not in segments:
                continue
            stats = segments[key].get("cancel_causes_stats", {})
            if stats:
                parts = []
                for cause, payload in sorted(
                    stats.items(),
                    key=lambda kv: kv[1].get("count", 0),
                    reverse=True,
                ):
                    count = payload.get("count", 0)
                    ratio = payload.get("ratio", 0.0)
                    parts.append(f"{cause}={count} ({ratio})")
                lines.append(f"- {key}: " + ", ".join(parts))
        lines.append("")
        lines.append("## Veto Streaks")
        for key in ("pre", "post", "all"):
            if key not in segments:
                continue
            streaks = segments[key].get("veto_streaks", {})
            if streaks:
                lines.append(
                    f"- {key}: max={streaks.get('max', 0)}, "
                    f"avg={streaks.get('avg', 0.0)}, "
                    f"accept_count={streaks.get('accept_count', 0)}"
                )
        lines.append("")
        lines.append("## Veto Streaks (Normalized)")
        for key in ("pre", "post", "all"):
            if key not in segments:
                continue
            streaks_norm = segments[key].get("veto_streaks_norm", {})
            if streaks_norm:
                lines.append(
                    f"- {key}: max={streaks_norm.get('max', 0.0)}, "
                    f"avg={streaks_norm.get('avg', 0.0)}"
                )
        lines.append("")
        lines.append("## Accept Contexts")
        for key in ("pre", "post", "all"):
            if key not in segments:
                continue
            contexts = segments[key].get("accept_contexts", [])
            if contexts:
                lines.append(f"- {key}: {contexts}")
        lines.append("")
    lines.append("## Nightly Audit (trace_v1)")
    if nightly:
        health = nightly.get("health", {})
        boundary = nightly.get("boundary", {})
        prospection = nightly.get("prospection", {})
        lines.append(f"- health_status: {health.get('status', 'unknown')}")
        lines.append(f"- boundary_span_max: {boundary.get('max_length', 0)}")
        lines.append(f"- boundary_span_chain_max: {boundary.get('span_chain_max', 0)}")
        lines.append(f"- prospection_reject_rate: {prospection.get('reject_rate', 0.0)}")
    else:
        audit_error = payload.get("nightly_audit_error")
        if audit_error:
            lines.append(f"- error: {audit_error}")
        else:
            lines.append("- error: trace_v1 audit missing")
    lines.append("")
    lines.append("## Recall Summary")
    lines.append(f"- anchors: {recall.get('anchors', {})}")
    lines.append(f"- confidence: {recall.get('confidence', {})}")
    lines.append("")
    lines.append("## Dream Prompt")
    lines.append(recall.get("dream_prompt", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=str, default=None, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--trace_root", type=str, default="trace_v1")
    ap.add_argument("--trace_log", type=str, default="logs/activation_traces.jsonl")
    ap.add_argument("--telemetry_log", type=str, default="")
    ap.add_argument("--audit_out", type=str, default="reports/audit")
    ap.add_argument("--out_json", type=str, default="")
    ap.add_argument("--out_md", type=str, default="")
    ap.add_argument("--deviant_days", type=int, default=3)
    ap.add_argument("--deviant_min_total", type=int, default=3)
    ap.add_argument("--deviant_ratio_threshold", type=float, default=0.15)
    args = ap.parse_args()

    if "<run_id>" in args.trace_root:
        raise SystemExit(
            "trace_root contains placeholder '<run_id>'. Replace it with the actual run_id."
        )

    day = date.fromisoformat(args.day) if args.day else date.today()
    runtime_cfg = load_runtime_cfg()
    telemetry_template = args.telemetry_log or runtime_cfg.telemetry.log_path
    trace_root = Path(args.trace_root)
    latest_day = _latest_trace_day(trace_root)
    if args.day is None and latest_day is not None:
        day = latest_day
    elif args.day is not None and latest_day is not None:
        if not (trace_root / day.isoformat()).exists():
            day = latest_day
    telemetry_path = _resolve_log_path(telemetry_template, day=day)

    report = generate_recall_report(args.trace_log, day=day)
    telemetry_entries = _load_telemetry([telemetry_path])
    kpi = _kpi_summary(telemetry_entries)

    trace_records = _load_trace_v1_events(trace_root, day)
    decision_records = _decision_cycle_records(trace_records)
    deviant_records = _deviant_records(trace_records)
    deviant_boundary_records = [
        record for record in deviant_records if _classify_deviant(record) == "boundary"
    ]
    resonance_records = [
        record for record in deviant_records if _classify_deviant(record) == "risk"
    ]
    transitions = _transition_events(trace_records)
    first_transition_ts = None
    transition_turn_index = None
    if transitions:
        first_transition_ts = transitions[0].get("timestamp_ms")
        transition_payload = transitions[0].get("transition", {})
        if isinstance(transition_payload, dict):
            transition_turn_index = transition_payload.get("transition_turn_index")
    pre_kpi = None
    post_kpi = None
    if isinstance(transition_turn_index, int):
        pre_kpi = _kpi_summary(telemetry_entries, end_step=transition_turn_index)
        post_kpi = _kpi_summary(telemetry_entries, start_step=transition_turn_index)

    trace_count = report.trace_count
    step_count = kpi.get("step_count", 0)
    coverage = (trace_count / step_count) if step_count else 0.0

    audit_payload: Dict[str, Any] | None = None
    audit_error: str | None = None
    try:
        audit_path = generate_audit(
            NightlyAuditConfig(
                trace_root=Path(args.trace_root),
                out_root=Path(args.audit_out),
                date_yyyy_mm_dd=day.isoformat(),
            )
        )
        audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        audit_error = f"trace_root missing: {exc}"

    payload: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(),
        "day": day.isoformat(),
        "telemetry_paths": [str(telemetry_path)],
        "trace_path": str(args.trace_log),
        "trace_root": str(args.trace_root),
        "nightly_audit": audit_payload,
        "nightly_audit_error": audit_error,
        "trace_count": trace_count,
        "trace_coverage": round(coverage, 3),
        "kpi": kpi,
        "transitions": {
            "count": len(transitions),
            "first_ts": first_transition_ts,
            "transition_turn_index": transition_turn_index,
            "pre_kpi": pre_kpi,
            "post_kpi": post_kpi,
        },
        "deviant": _deviant_summary(deviant_boundary_records, decision_records),
        "resonance": _deviant_summary(resonance_records, decision_records),
        "executed_boundary": _executed_boundary_stats(decision_records),
        "executed_boundary_by_world": _executed_boundary_by_world(decision_records),
        "decision_boundary": _decision_boundary_stats(decision_records),
        "decision_boundary_by_world": _decision_boundary_by_world(decision_records),
        "decision_score": _decision_score_stats(decision_records),
        "decision_score_by_world": _decision_score_by_world(decision_records),
        "high_boundary_cancel": _high_boundary_cancel_stats(decision_records),
        "high_boundary_cancel_by_world": _high_boundary_cancel_by_world(decision_records),
        "cancel_summary": _cancel_reason_summary(decision_records),
        "cancel_by_world": _cancel_reason_by_world(decision_records),
        "ru_v0_summary": _ru_v0_summary(trace_records),
        "segments": {},
        "recall_report": report.to_dict(),
    }
    deviant_by_world: Dict[str, int] = {}
    for record in deviant_boundary_records:
        world_type = record.get("world_type") or "unknown"
        deviant_by_world[world_type] = deviant_by_world.get(world_type, 0) + 1
    world_breakdown: Dict[str, Dict[str, Any]] = {}
    executed_by_world = payload.get("executed_boundary_by_world", {})
    decision_by_world = payload.get("decision_boundary_by_world", {})
    score_by_world = payload.get("decision_score_by_world", {})
    high_boundary_by_world = payload.get("high_boundary_cancel_by_world", {})
    cancel_by_world = payload.get("cancel_by_world", {})
    all_worlds = (
        set(executed_by_world.keys())
        | set(cancel_by_world.keys())
        | set(deviant_by_world.keys())
        | set(decision_by_world.keys())
        | set(score_by_world.keys())
        | set(high_boundary_by_world.keys())
    )
    for world_type in sorted(all_worlds):
        executed = executed_by_world.get(world_type, {})
        decision = decision_by_world.get(world_type, {})
        score = score_by_world.get(world_type, {})
        high_boundary = high_boundary_by_world.get(world_type, {})
        cancel = cancel_by_world.get(world_type, {})
        world_breakdown[world_type] = {
            "executed_count": executed.get("executed_count", 0),
            "executed_boundary_max": executed.get("executed_boundary_max", 0.0),
            "all_decision_boundary_max": decision.get("all_decision_boundary_max", 0.0),
            "cancel_boundary_max": decision.get("cancel_boundary_max", 0.0),
            "high_boundary_cancel_count": high_boundary.get("count", 0),
            "decision_score_executed_max": score.get("decision_score_executed_max", 0.0),
            "decision_score_cancel_max": score.get("decision_score_cancel_max", 0.0),
            "u_hat_cancel_max": score.get("u_hat_cancel_max", 0.0),
            "veto_score_cancel_max": score.get("veto_score_cancel_max", 0.0),
            "deviant_count": deviant_by_world.get(world_type, 0),
            "cancel_total": cancel.get("total", 0),
            "cancel_reasons": cancel.get("reasons", {}),
        }
    payload["world_breakdown"] = world_breakdown

    segments = _segment_telemetry(telemetry_entries, transition_turn_index if isinstance(transition_turn_index, int) else None)
    segment_payload: Dict[str, Any] = {}
    accept_contexts = _accept_contexts(decision_records, transition_turn_index if isinstance(transition_turn_index, int) else None)
    for key, events in segments.items():
        summary = _kpi_summary(events)
        steps = summary.get("step_count", 0) or 0
        segment_payload[key] = {
            "steps": steps,
            "cancel_rate": summary.get("cancel_rate", 0.0),
            "cancel_causes_stats": _cancel_causes_stats(events),
            "veto_streaks": _veto_streak_stats(events),
            "veto_streaks_norm": {},
            "mean_drive": _mean_field(events, "drive"),
            "mean_uncertainty": _mean_field(events, "uncertainty"),
            "accept_contexts": accept_contexts.get(key, []),
        }
        streaks = segment_payload[key].get("veto_streaks", {})
        if steps and streaks:
            segment_payload[key]["veto_streaks_norm"] = {
                "max": round(streaks.get("max", 0) / steps, 3),
                "avg": round(streaks.get("avg", 0.0) / steps, 3),
            }
    payload["segments"] = segment_payload
    window_days = _recent_days(trace_root, day, int(args.deviant_days))
    window_records = _load_trace_v1_for_days(trace_root, window_days)
    window_decisions = _decision_cycle_records(window_records)
    window_deviants = _deviant_records(window_records)
    window_boundary = [r for r in window_deviants if _classify_deviant(r) == "boundary"]
    window_resonance = [r for r in window_deviants if _classify_deviant(r) == "risk"]
    window_stats = _deviant_window_stats_for_records(window_boundary, window_decisions)
    proposals: List[Dict[str, Any]] = []
    for world_type, stats in window_stats.items():
        deviant_count = int(stats.get("deviant_count", 0))
        decision_count = int(stats.get("decision_count", 0))
        ratio = float(stats.get("ratio", 0.0))
        if deviant_count >= int(args.deviant_min_total) and ratio >= float(args.deviant_ratio_threshold):
            proposals.append(
                {
                    "world_type": world_type,
                    "deviant_count": deviant_count,
                    "decision_count": decision_count,
                    "ratio": round(ratio, 3),
                    "window_days": [d.isoformat() for d in window_days],
                }
            )
    payload["deviant_window"] = {
        "days": [d.isoformat() for d in window_days],
        "by_world": window_stats,
    }
    payload["world_prior_proposals"] = proposals
    resonance_stats = _deviant_window_stats_for_records(window_resonance, window_decisions)
    resonance_notices: List[Dict[str, Any]] = []
    for world_type, stats in resonance_stats.items():
        deviant_count = int(stats.get("deviant_count", 0))
        decision_count = int(stats.get("decision_count", 0))
        ratio = float(stats.get("ratio", 0.0))
        if deviant_count >= int(args.deviant_min_total) and ratio >= float(args.deviant_ratio_threshold):
            resonance_notices.append(
                {
                    "world_type": world_type,
                    "deviant_count": deviant_count,
                    "decision_count": decision_count,
                    "ratio": round(ratio, 3),
                    "window_days": [d.isoformat() for d in window_days],
                }
            )
    payload["resonance_window"] = {
        "days": [d.isoformat() for d in window_days],
        "by_world": resonance_stats,
    }
    payload["resonance_notices"] = resonance_notices

    stamp = day.strftime("%Y%m%d")
    out_json = Path(args.out_json) if args.out_json else Path(f"reports/nightly_audit_{stamp}.json")
    out_md = Path(args.out_md) if args.out_md else Path(f"reports/nightly_audit_{stamp}.md")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(out_md, payload)

    print(f"[info] nightly report: {out_json}")
    print(f"[info] nightly markdown: {out_md}")


if __name__ == "__main__":
    main()
