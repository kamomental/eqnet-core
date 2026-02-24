#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Trace replay CLI for deterministic what-if evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from eqnet.runtime.replay.diff import build_diff_summary, load_diff_ranking_policy
from eqnet.runtime.replay.pipeline import ReplayConfig, run_replay
from eqnet.runtime.replay.report_writer import write_diff_reports, write_replay_reports


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay trace logs and optionally compare two config sets.")
    ap.add_argument("--trace-path", required=True, type=str, help="Path to trace jsonl file or directory.")
    ap.add_argument("--start-day-key", type=str, default=None, help="Inclusive day key YYYY-MM-DD.")
    ap.add_argument("--end-day-key", type=str, default=None, help="Inclusive day key YYYY-MM-DD.")
    ap.add_argument("--config-set", type=str, default=None, help="Primary config set under configs/config_sets/<name>.")
    ap.add_argument("--config-set-b", type=str, default=None, help="Optional secondary config set for what-if diff.")
    ap.add_argument("--config-root", type=str, default="configs", help="Root directory for config files.")
    ap.add_argument("--out-dir", type=str, default="reports/replay", help="Replay output directory.")
    ap.add_argument("--format", choices=["jsonl", "md"], default="jsonl", help="Preferred human-readable format.")
    args = ap.parse_args()

    trace_path = Path(args.trace_path)
    out_dir = Path(args.out_dir)
    cfg_a = ReplayConfig(
        trace_path=trace_path,
        start_day_key=args.start_day_key,
        end_day_key=args.end_day_key,
        config_set=args.config_set,
        config_root=Path(args.config_root),
    )
    run_a = run_replay(cfg_a)
    write_replay_reports(out_dir=out_dir, summary=run_a, daily_rows=list(run_a.get("daily") or []))

    if args.config_set_b:
        cfg_b = ReplayConfig(
            trace_path=trace_path,
            start_day_key=args.start_day_key,
            end_day_key=args.end_day_key,
            config_set=args.config_set_b,
            config_root=Path(args.config_root),
        )
        run_b = run_replay(cfg_b)
        out_dir_b = out_dir / f"set_{args.config_set_b}"
        write_replay_reports(out_dir=out_dir_b, summary=run_b, daily_rows=list(run_b.get("daily") or []))
        ranking_policy, ranking_policy_meta = load_diff_ranking_policy(Path(args.config_root) / "diff_ranking_policy_v0.yaml")
        diff = build_diff_summary(
            run_a,
            run_b,
            comparison_scope={
                "trace_path": str(trace_path.as_posix()),
                "start_day_key": args.start_day_key,
                "end_day_key": args.end_day_key,
                "config_set_a": args.config_set,
                "config_set_b": args.config_set_b,
            },
            ranking_policy=ranking_policy,
            ranking_policy_meta=ranking_policy_meta,
        )
        write_diff_reports(out_dir=out_dir, diff_summary=diff)


if __name__ == "__main__":
    main()
