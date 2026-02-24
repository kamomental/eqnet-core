#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate sync micro outcomes and apply realtime downshift when needed."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from eqnet.runtime.future_contracts import load_sync_quality_rules
from eqnet.runtime.sync_realtime import (
    evaluate_downshift_state,
    evaluate_sync_micro_outcome,
    load_realtime_downshift_policy,
)
from eqnet.telemetry.sync_downshift_writer import SyncDownshiftWriter, SyncDownshiftWriterConfig
from eqnet.telemetry.sync_micro_outcome_writer import SyncMicroOutcomeWriter, SyncMicroOutcomeWriterConfig


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except Exception:
            continue
        if isinstance(payload, dict):
            out.append(payload)
    return out


def _collect(telemetry_dir: Path, pattern: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(telemetry_dir.glob(pattern)):
        rows.extend(_read_jsonl(path))
    return rows


def _index_signals(path: Path | None) -> Dict[str, float]:
    if path is None or not path.exists():
        return {}
    out: Dict[str, float] = {}
    for row in _read_jsonl(path):
        eid = str(row.get("execution_id") or "")
        observed = row.get("observed_r")
        if eid and isinstance(observed, (int, float)):
            out[eid] = float(observed)
    return out


def _source_week(ts_ms: int) -> str:
    dt = datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc)
    y, w, _ = dt.isocalendar()
    return f"{y}-W{w:02d}"


def _seen_outcomes(rows: List[Mapping[str, Any]]) -> set[Tuple[str, int]]:
    seen: set[Tuple[str, int]] = set()
    for row in rows:
        eid = str(row.get("execution_id") or "")
        window = int(row.get("window_sec") or 0)
        if eid and window > 0:
            seen.add((eid, window))
    return seen


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate sync micro outcomes and optional downshift.")
    ap.add_argument("--telemetry-dir", default="telemetry", type=str)
    ap.add_argument("--sync-quality-rules", default="configs/sync_quality_rules_v0.yaml", type=str)
    ap.add_argument("--downshift-policy", default="configs/realtime_downshift_policy_v0.yaml", type=str)
    ap.add_argument("--signals-jsonl", default=None, type=str, help="Optional observed_r stream keyed by execution_id.")
    ap.add_argument("--window-sec", default=60, type=int)
    ap.add_argument("--now-ts-ms", default=None, type=int)
    ap.add_argument("--apply-downshift", action="store_true")
    args = ap.parse_args()

    telemetry_dir = Path(args.telemetry_dir)
    now_ts_ms = int(args.now_ts_ms) if args.now_ts_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000)
    window_sec = int(args.window_sec)
    rules = load_sync_quality_rules(Path(args.sync_quality_rules))
    signals = _index_signals(Path(args.signals_jsonl)) if args.signals_jsonl else {}

    executions = _collect(telemetry_dir, "sync_cue_executions-*.jsonl")
    outcomes_existing = _collect(telemetry_dir, "sync_micro_outcomes-*.jsonl")
    seen = _seen_outcomes(outcomes_existing)
    outcome_writer = SyncMicroOutcomeWriter(SyncMicroOutcomeWriterConfig(telemetry_dir=telemetry_dir))

    written = 0
    for row in executions:
        execution_id = str(row.get("execution_id") or "")
        proposal_id = str(row.get("proposal_id") or "")
        if not execution_id or not proposal_id:
            continue
        key = (execution_id, window_sec)
        if key in seen:
            continue
        executed_at = int(row.get("timestamp_ms") or 0)
        eval_at = int(executed_at + window_sec * 1000)
        if eval_at > now_ts_ms:
            continue
        baseline = row.get("sync_order_parameter_r")
        baseline_r = float(baseline) if isinstance(baseline, (int, float)) else None
        observed_r = signals.get(execution_id)
        eval_result = evaluate_sync_micro_outcome(
            baseline_r=baseline_r,
            observed_r=observed_r,
            window_sec=window_sec,
            evaluated_at_eval_ts_ms=eval_at,
            rules=rules,
        )
        outcome_writer.append(
            timestamp_ms=now_ts_ms,
            proposal_id=proposal_id,
            execution_id=execution_id,
            window_sec=window_sec,
            baseline_r=baseline_r,
            observed_r=observed_r,
            delta_r=eval_result.get("delta_r"),
            result=str(eval_result.get("result") or "UNKNOWN"),
            reason_codes=list(eval_result.get("reason_codes") or []),
            evaluated_at_eval_ts_ms=eval_at,
            source_week=_source_week(now_ts_ms),
            extra={"origin_channel": row.get("origin_channel")},
        )
        written += 1

    downshift_written = 0
    if args.apply_downshift:
        policy = load_realtime_downshift_policy(Path(args.downshift_policy))
        outcomes_all = _collect(telemetry_dir, "sync_micro_outcomes-*.jsonl")
        state = evaluate_downshift_state(outcomes=outcomes_all, now_ts_ms=now_ts_ms, policy=policy)
        if bool(state.get("applied")):
            policy_meta = {
                "policy_version": str(policy.get("policy_version") or "realtime_downshift_policy_v0"),
                "policy_source": str(policy.get("policy_source") or "configs/realtime_downshift_policy_v0.yaml"),
            }
            writer = SyncDownshiftWriter(SyncDownshiftWriterConfig(telemetry_dir=telemetry_dir))
            writer.append(
                timestamp_ms=now_ts_ms,
                reason_codes=list(state.get("reason_codes") or []),
                cooldown_until_ts_ms=int(state.get("cooldown_until_ts_ms") or now_ts_ms),
                actions=list(state.get("actions") or []),
                policy_meta=policy_meta,
                source_week=_source_week(now_ts_ms),
            )
            downshift_written = 1

    print(json.dumps({"ok": True, "micro_outcomes_written": written, "downshift_written": downshift_written}, ensure_ascii=False))


if __name__ == "__main__":
    main()
