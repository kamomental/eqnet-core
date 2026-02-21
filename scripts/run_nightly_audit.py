#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate a nightly audit report from activation traces and telemetry."""

from __future__ import annotations

import argparse
import json
import os
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


def _extract_nightly_metrics(report_payload: Dict[str, Any]) -> Dict[str, float]:
    nightly = report_payload.get("nightly_audit") or {}
    if not isinstance(nightly, dict):
        return {"sat_p95": 0.0, "low_ratio": 0.0}
    uncertainty = nightly.get("uncertainty_confidence") or {}
    low_ratio = 0.0
    if isinstance(uncertainty, dict):
        try:
            low_ratio = float(uncertainty.get("low_ratio", 0.0) or 0.0)
        except (TypeError, ValueError):
            low_ratio = 0.0
    try:
        sat_p95 = float(nightly.get("lazy_rag_sat_ratio_p95", 0.0) or 0.0)
    except (TypeError, ValueError):
        sat_p95 = 0.0
    return {"sat_p95": sat_p95, "low_ratio": low_ratio}


def _load_previous_report(out_json: Path, day: date) -> Dict[str, Any]:
    prev_day = day.fromordinal(day.toordinal() - 1)
    prev_name = f"nightly_audit_{prev_day.strftime('%Y%m%d')}.json"
    prev_path = out_json.parent / prev_name
    if not prev_path.exists():
        return {}
    try:
        return json.loads(prev_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _week_day_range(day: date) -> List[date]:
    weekday = day.weekday()
    monday = day.fromordinal(day.toordinal() - weekday)
    return [monday.fromordinal(monday.toordinal() + offset) for offset in range(7)]


def _weekly_metric_summary(
    out_dir: Path,
    day: date,
    current_report: Dict[str, Any],
) -> Dict[str, float]:
    sat_values: List[float] = []
    low_values: List[float] = []
    current_metrics = _extract_nightly_metrics(current_report)
    sat_values.append(current_metrics["sat_p95"])
    low_values.append(current_metrics["low_ratio"])
    for candidate in _week_day_range(day):
        if candidate == day:
            continue
        candidate_name = f"nightly_audit_{candidate.strftime('%Y%m%d')}.json"
        candidate_path = out_dir / candidate_name
        if not candidate_path.exists():
            continue
        try:
            payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        metrics = _extract_nightly_metrics(payload)
        sat_values.append(metrics["sat_p95"])
        low_values.append(metrics["low_ratio"])
    count = len(sat_values)
    sat_avg = (sum(sat_values) / count) if count else 0.0
    low_avg = (sum(low_values) / count) if count else 0.0
    return {
        "count": float(count),
        "sat_p95_avg": round(sat_avg, 3),
        "sat_p95_max": round(max(sat_values) if sat_values else 0.0, 3),
        "low_ratio_avg": round(low_avg, 3),
        "low_ratio_max": round(max(low_values) if low_values else 0.0, 3),
    }


def _weekly_snapshot(runtime_cfg: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    rag_assoc = getattr(getattr(runtime_cfg, "rag", None), "assoc_score", None)
    rag_weights = getattr(rag_assoc, "weights", None)
    rag_clamp = getattr(rag_assoc, "clamp", None)
    ui_cfg = getattr(runtime_cfg, "ui", None)
    return {
        "sat_warn_threshold": float(payload.get("lazy_rag_sat_ratio_alert_threshold", 0.6) or 0.6),
        "confidence_low_max": float(getattr(ui_cfg, "uncertainty_confidence_low_max", 0.54) or 0.54),
        "confidence_mid_max": float(getattr(ui_cfg, "uncertainty_confidence_mid_max", 0.79) or 0.79),
        "assoc_enabled": bool(getattr(rag_assoc, "enabled", False)),
        "assoc_normalize_weights": bool(getattr(rag_assoc, "normalize_weights", True)),
        "assoc_temporal_tau_sec": float(getattr(rag_assoc, "temporal_tau_sec", 86400.0) or 86400.0),
        "assoc_weights": {
            "semantic": float(getattr(rag_weights, "semantic", 1.0) or 1.0),
            "temporal": float(getattr(rag_weights, "temporal", 0.1) or 0.1),
            "affective": float(getattr(rag_weights, "affective", 0.12) or 0.12),
            "value": float(getattr(rag_weights, "value", 0.15) or 0.15),
            "open_loop": float(getattr(rag_weights, "open_loop", 0.08) or 0.08),
        },
        "assoc_clamp": {
            "min": float(getattr(rag_clamp, "min", -5.0) or -5.0),
            "max": float(getattr(rag_clamp, "max", 5.0) or 5.0),
        },
    }


def _flatten_snapshot(snapshot: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in snapshot.items():
        joined = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_snapshot(value, joined))
        else:
            flat[joined] = value
    return flat


def _snapshot_changed_keys(current: Dict[str, Any], previous: Dict[str, Any]) -> List[str]:
    cur_flat = _flatten_snapshot(current)
    prev_flat = _flatten_snapshot(previous)
    keys = set(cur_flat.keys()) | set(prev_flat.keys())
    changed = [key for key in sorted(keys) if cur_flat.get(key) != prev_flat.get(key)]
    return changed


def _recommended_action_code(
    weekly_metrics: Dict[str, float],
    payload: Dict[str, Any],
) -> str:
    sample_count_raw = weekly_metrics.get("count")
    if sample_count_raw is not None:
        sample_count = float(sample_count_raw or 0.0)
        if sample_count <= 0.0:
            return "none"
    sat_avg = float(weekly_metrics.get("sat_p95_avg", 0.0) or 0.0)
    sat_max = float(weekly_metrics.get("sat_p95_max", 0.0) or 0.0)
    low_max = float(weekly_metrics.get("low_ratio_max", 0.0) or 0.0)
    nightly = payload.get("nightly_audit") or {}
    top3 = nightly.get("uncertainty_reason_top3") if isinstance(nightly, dict) else []
    reasons: List[str] = []
    if isinstance(top3, list):
        for item in top3:
            if isinstance(item, dict):
                reason = item.get("reason")
                if isinstance(reason, str):
                    reasons.append(reason)
    if sat_avg >= 0.6 and sat_max >= 0.7:
        return "saturation_high"
    if low_max >= 0.25 and "retrieval_sparse" in reasons:
        return "retrieval_sparse"
    if low_max >= 0.25:
        return "uncertainty_high"
    return "stable"


def _load_previous_weekly_snapshot(weekly_json_path: Path, day: date) -> Dict[str, Any]:
    prev_week_day = day.fromordinal(day.toordinal() - 7)
    prev_year, prev_week, _ = prev_week_day.isocalendar()
    prev_path = weekly_json_path.parent / f"weekly_calibration_{prev_year}-W{prev_week:02d}.json"
    if not prev_path.exists():
        return {}
    try:
        payload = json.loads(prev_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    snapshot = payload.get("snapshot")
    if isinstance(snapshot, dict):
        return snapshot
    return {}


def _write_weekly_calibration(
    path: Path,
    *,
    day: date,
    snapshot: Dict[str, Any],
    weekly_metrics: Dict[str, float],
    changed_keys: List[str],
    recommended_action_code: str,
    recommended_action_text: str,
    recommendation_evidence: str,
    proposal_title: str,
    evidence_label: str,
    changed_keys_max: int,
    changed_keys_more_template: str,
) -> None:
    iso_year, iso_week, _ = day.isocalendar()
    assoc_weights = snapshot.get("assoc_weights", {}) if isinstance(snapshot.get("assoc_weights"), dict) else {}
    assoc_clamp = snapshot.get("assoc_clamp", {}) if isinstance(snapshot.get("assoc_clamp"), dict) else {}

    lines: List[str] = []
    lines.append(f"# Weekly Calibration ({iso_year}-W{iso_week:02d})")
    lines.append("")
    lines.append(f"- generated_at: {datetime.utcnow().isoformat()}")
    lines.append(f"- day: {day.isoformat()}")
    lines.append("")
    lines.append("## Weekly Metrics")
    lines.append(
        f"- sat_p95 avg/max: {weekly_metrics.get('sat_p95_avg', 0.0)} / {weekly_metrics.get('sat_p95_max', 0.0)}"
    )
    lines.append(
        f"- low_ratio avg/max: {weekly_metrics.get('low_ratio_avg', 0.0)} / {weekly_metrics.get('low_ratio_max', 0.0)}"
    )
    lines.append(f"- samples: {int(weekly_metrics.get('count', 0.0))}")
    lines.append("")
    lines.append("## Thresholds")
    lines.append(f"- sat_warn_threshold: {snapshot.get('sat_warn_threshold', 0.6)}")
    lines.append(f"- confidence_low_max: {snapshot.get('confidence_low_max', 0.54)}")
    lines.append(f"- confidence_mid_max: {snapshot.get('confidence_mid_max', 0.79)}")
    lines.append("")
    lines.append("## LazyRAG / RagRetriever")
    lines.append(f"- assoc_enabled: {bool(snapshot.get('assoc_enabled', False))}")
    lines.append(f"- assoc_normalize_weights: {bool(snapshot.get('assoc_normalize_weights', True))}")
    lines.append(f"- assoc_temporal_tau_sec: {float(snapshot.get('assoc_temporal_tau_sec', 86400.0) or 86400.0)}")
    lines.append(
        "- assoc_weights: "
        f"semantic={float(assoc_weights.get('semantic', 1.0) or 1.0)}, "
        f"temporal={float(assoc_weights.get('temporal', 0.1) or 0.1)}, "
        f"affective={float(assoc_weights.get('affective', 0.12) or 0.12)}, "
        f"value={float(assoc_weights.get('value', 0.15) or 0.15)}, "
        f"open_loop={float(assoc_weights.get('open_loop', 0.08) or 0.08)}"
    )
    lines.append(
        "- assoc_clamp: "
        f"min={float(assoc_clamp.get('min', -5.0) or -5.0)}, "
        f"max={float(assoc_clamp.get('max', 5.0) or 5.0)}"
    )
    lines.append("")
    lines.append(f"## {proposal_title}")
    lines.append(f"- {recommended_action_text} [{recommended_action_code}]")
    lines.append(f"- {evidence_label}: {recommendation_evidence}")
    lines.append("")
    lines.append("## Changed Keys (vs previous week)")
    if changed_keys:
        display = changed_keys[: max(1, int(changed_keys_max))]
        lines.append(f"- {', '.join(display)}")
        remains = len(changed_keys) - len(display)
        if remains > 0:
            lines.append(f"- {changed_keys_more_template.format(count=remains)}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Notes")
    lines.append("- ")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


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
    qualia_gate_stats = (nightly.get("qualia_gate_stats") or {}) if isinstance(nightly, dict) else {}
    memory_hint = qualia_gate_stats.get("memory_hint") if isinstance(qualia_gate_stats, dict) else {}
    if isinstance(memory_hint, dict) and memory_hint:
        lines.append("## Memory Hint")
        lines.append(f"- rate: {memory_hint.get('memory_hint_rate', 0.0)}")
        lines.append(f"- blocked_rate: {memory_hint.get('memory_hint_blocked_rate', 0.0)}")
        lines.append(f"- blocked_reasons: {memory_hint.get('memory_hint_blocked_reason_topk', [])}")
        lines.append(f"- key_topk: {memory_hint.get('memory_hint_key_topk', [])}")
        lines.append(f"- category_topk: {memory_hint.get('memory_hint_category_topk', [])}")
        lines.append(f"- avg_interrupt_cost_when_blocked: {memory_hint.get('avg_interrupt_cost_when_blocked', 0.0)}")
        lines.append(f"- community_turn_violation_count: {memory_hint.get('community_turn_violation_count', 0)}")
        lines.append(
            f"- pressure_mean: {memory_hint.get('memory_hint_pressure_mean', 0.0)}"
        )
        lines.append(f"- pressure_p95: {memory_hint.get('memory_hint_pressure_p95', 0.0)}")
        lines.append(
            f"- pressure_delta_mean: {memory_hint.get('memory_hint_pressure_delta_mean', 0.0)}"
        )
        lines.append(
            f"- blocked_pressure_mean: {memory_hint.get('memory_hint_blocked_pressure_mean', 0.0)}"
        )
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
        sat_ratio_p95 = float(nightly.get("lazy_rag_sat_ratio_p95", 0.0) or 0.0)
        sat_ratio_count = int(nightly.get("lazy_rag_sat_ratio_count", 0) or 0)
        uncertainty_top3 = nightly.get("uncertainty_reason_top3", [])
        uncertainty_conf = nightly.get("uncertainty_confidence", {}) if isinstance(nightly.get("uncertainty_confidence"), dict) else {}
        reason_labels = payload.get("uncertainty_reason_labels", {})
        if not isinstance(reason_labels, dict):
            reason_labels = {}
        sat_alert_threshold = float(payload.get("lazy_rag_sat_ratio_alert_threshold", 0.6) or 0.6)
        lines.append(f"- health_status: {health.get('status', 'unknown')}")
        lines.append(f"- boundary_span_max: {boundary.get('max_length', 0)}")
        lines.append(f"- boundary_span_chain_max: {boundary.get('span_chain_max', 0)}")
        lines.append(f"- prospection_reject_rate: {prospection.get('reject_rate', 0.0)}")
        lines.append(f"- lazy_rag_sat_ratio_p95: {sat_ratio_p95}")
        lines.append(f"- lazy_rag_sat_ratio_count: {sat_ratio_count}")
        if sat_ratio_count > 0 and sat_ratio_p95 >= sat_alert_threshold:
            lines.append(
                f"- WARNING: LazyRAG score saturation high (p95={sat_ratio_p95:.3f}, "
                f"count={sat_ratio_count}, threshold={sat_alert_threshold:.3f})"
            )
        sat_delta = payload.get("lazy_rag_sat_ratio_p95_delta")
        low_delta = payload.get("confidence_low_ratio_delta")
        prev_sat = payload.get("prev_lazy_rag_sat_ratio_p95")
        prev_low = payload.get("prev_confidence_low_ratio")
        if isinstance(sat_delta, (int, float)) and isinstance(low_delta, (int, float)):
            lines.append(
                f"- delta: sat_p95={float(sat_delta):+0.3f} / low_ratio={float(low_delta):+0.3f} "
                f"(prev: sat_p95={float(prev_sat or 0.0):0.3f}, low_ratio={float(prev_low or 0.0):0.3f})"
            )
        if isinstance(uncertainty_top3, list) and uncertainty_top3:
            parts: List[str] = []
            for item in uncertainty_top3:
                if not isinstance(item, dict):
                    continue
                reason = str(item.get("reason", "unknown"))
                reason_label = str(reason_labels.get(reason, reason))
                count = int(item.get("count", 0) or 0)
                if reason_label == reason:
                    parts.append(f"{reason}({count})")
                else:
                    parts.append(f"{reason_label}[{reason}]({count})")
            if parts:
                lines.append(f"- uncertainty_reason_top3: {', '.join(parts)}")
        if uncertainty_conf:
            total = int(uncertainty_conf.get("total", 0) or 0)
            low = int(uncertainty_conf.get("low", 0) or 0)
            mid = int(uncertainty_conf.get("mid", 0) or 0)
            high = int(uncertainty_conf.get("high", 0) or 0)
            low_ratio = float(uncertainty_conf.get("low_ratio", 0.0) or 0.0)
            lines.append(
                f"- confidence_low_ratio: {low_ratio:.3f} "
                f"(low={low}, mid={mid}, high={high}, total={total})"
            )
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

    sat_alert_threshold = float(
        os.getenv("EQNET_LAZY_RAG_SCORE_DIAG_WARN_SAT_RATIO", "0.6")
    )

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
        "lazy_rag_sat_ratio_alert_threshold": sat_alert_threshold,
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

    current_metrics = _extract_nightly_metrics({"nightly_audit": audit_payload or {}})
    prev_report = _load_previous_report(out_json, day)
    prev_metrics = _extract_nightly_metrics(prev_report) if prev_report else {"sat_p95": 0.0, "low_ratio": 0.0}
    payload["lazy_rag_sat_ratio_p95_delta"] = round(current_metrics["sat_p95"] - prev_metrics["sat_p95"], 3)
    payload["confidence_low_ratio_delta"] = round(current_metrics["low_ratio"] - prev_metrics["low_ratio"], 3)
    payload["prev_lazy_rag_sat_ratio_p95"] = round(prev_metrics["sat_p95"], 3)
    payload["prev_confidence_low_ratio"] = round(prev_metrics["low_ratio"], 3)
    ui_cfg = getattr(runtime_cfg, "ui", None)
    reason_labels_cfg = getattr(ui_cfg, "uncertainty_reason_labels", {}) if ui_cfg else {}
    payload["uncertainty_reason_labels"] = dict(reason_labels_cfg) if isinstance(reason_labels_cfg, dict) else {}

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(out_md, payload)
    iso_year, iso_week, _ = day.isocalendar()
    weekly_path = out_json.parent / f"weekly_calibration_{iso_year}-W{iso_week:02d}.md"
    weekly_json_path = out_json.parent / f"weekly_calibration_{iso_year}-W{iso_week:02d}.json"
    weekly_metrics = _weekly_metric_summary(out_json.parent, day, payload)
    snapshot = _weekly_snapshot(runtime_cfg, payload)
    previous_snapshot = _load_previous_weekly_snapshot(weekly_json_path, day)
    changed_keys = _snapshot_changed_keys(snapshot, previous_snapshot)
    recommended_action_code = _recommended_action_code(weekly_metrics, payload)
    sat_avg = float(weekly_metrics.get("sat_p95_avg", 0.0) or 0.0)
    sat_max = float(weekly_metrics.get("sat_p95_max", 0.0) or 0.0)
    low_avg = float(weekly_metrics.get("low_ratio_avg", 0.0) or 0.0)
    low_max = float(weekly_metrics.get("low_ratio_max", 0.0) or 0.0)
    recommendation_evidence = (
        f"sat_p95_avg={sat_avg:.3f}, sat_p95_max={sat_max:.3f}, "
        f"low_ratio_avg={low_avg:.3f}, low_ratio_max={low_max:.3f}"
    )
    ui_cfg = getattr(runtime_cfg, "ui", None)
    action_labels_cfg = getattr(ui_cfg, "weekly_action_labels", {}) if ui_cfg else {}
    if not isinstance(action_labels_cfg, dict):
        action_labels_cfg = {}
    default_action_labels = {
        "none": "大きな見直しは不要そうです",
        "saturation_high": "飽和傾向が高いため、境界幅・注意配分・時間感度の見直しを検討",
        "retrieval_sparse": "根拠不足傾向のため、想起候補の裾野と参照条件の見直しを検討",
        "uncertainty_high": "不確実性分布が高いため、上位理由に沿って境界条件の見直しを検討",
        "stable": "現状は安定。設定を維持しつつ観測メモを継続",
    }
    action_labels = {**default_action_labels, **action_labels_cfg}
    recommended_action_text = str(action_labels.get(recommended_action_code, action_labels["stable"]))
    proposal_title = str(getattr(ui_cfg, "weekly_proposal_title", "次の見直し候補")) if ui_cfg else "次の見直し候補"
    evidence_label = str(getattr(ui_cfg, "weekly_evidence_label", "根拠")) if ui_cfg else "根拠"
    changed_keys_max = int(getattr(ui_cfg, "weekly_changed_keys_max", 12) or 12) if ui_cfg else 12
    changed_keys_more_template = (
        str(getattr(ui_cfg, "weekly_changed_keys_more_template", "+{count}件"))
        if ui_cfg
        else "+{count}件"
    )
    _write_weekly_calibration(
        weekly_path,
        day=day,
        snapshot=snapshot,
        weekly_metrics=weekly_metrics,
        changed_keys=changed_keys,
        recommended_action_code=recommended_action_code,
        recommended_action_text=recommended_action_text,
        recommendation_evidence=recommendation_evidence,
        proposal_title=proposal_title,
        evidence_label=evidence_label,
        changed_keys_max=changed_keys_max,
        changed_keys_more_template=changed_keys_more_template,
    )
    weekly_payload = {
        "schema_version": 1,
        "generated_at": datetime.utcnow().isoformat(),
        "week": f"{iso_year}-W{iso_week:02d}",
        "day": day.isoformat(),
        "metrics": weekly_metrics,
        "snapshot": snapshot,
        "changed_keys": changed_keys,
        "recommended_action_code": recommended_action_code,
        "recommended_action": recommended_action_text,
        "recommendation_evidence": recommendation_evidence,
    }
    weekly_json_path.write_text(
        json.dumps(weekly_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[info] nightly report: {out_json}")
    print(f"[info] nightly markdown: {out_md}")
    print(f"[info] weekly calibration: {weekly_path}")


if __name__ == "__main__":
    main()
