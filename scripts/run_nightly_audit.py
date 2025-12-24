#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate a nightly audit report from activation traces and telemetry."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

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
        "segments": {},
        "recall_report": report.to_dict(),
    }

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
