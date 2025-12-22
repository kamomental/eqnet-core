#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate a nightly audit report from activation traces and telemetry."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from runtime.nightly_report import generate_recall_report
from runtime.config import load_runtime_cfg
from eqnet.telemetry.nightly_audit import NightlyAuditConfig, generate_audit


def _resolve_log_path(template: str, *, day: date) -> Path:
    if "%" not in template:
        return Path(template)
    return Path(day.strftime(template))


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


def _kpi_summary(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    steps = [e for e in events if e.get("event") == "minimal_heartos.step"]
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
    lines.append("## Nightly Audit (trace_v1)")
    if nightly:
        health = nightly.get("health", {})
        boundary = nightly.get("boundary", {})
        prospection = nightly.get("prospection", {})
        lines.append(f"- health_status: {health.get('status', 'unknown')}")
        lines.append(f"- boundary_span_max: {boundary.get('max_length', 0)}")
        lines.append(f"- prospection_reject_rate: {prospection.get('reject_rate', 0.0)}")
    else:
        lines.append(f"- error: {payload.get('nightly_audit_error')}")
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

    day = date.fromisoformat(args.day) if args.day else date.today()
    runtime_cfg = load_runtime_cfg()
    telemetry_template = args.telemetry_log or runtime_cfg.telemetry.log_path
    telemetry_path = _resolve_log_path(telemetry_template, day=day)

    report = generate_recall_report(args.trace_log, day=day)
    telemetry_entries = _load_telemetry([telemetry_path])
    kpi = _kpi_summary(telemetry_entries)

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
        "recall_report": report.to_dict(),
    }

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
