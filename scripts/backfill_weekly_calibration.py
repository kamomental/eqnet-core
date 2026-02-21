#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backfill weekly calibration artifacts from trace day directories."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import List, Tuple


def _discover_days(trace_root: Path) -> List[date]:
    days: List[date] = []
    if not trace_root.exists():
        return days
    for child in trace_root.iterdir():
        if not child.is_dir():
            continue
        try:
            days.append(date.fromisoformat(child.name))
        except ValueError:
            continue
    days.sort()
    return days


def _weekly_json_path(reports_dir: Path, day: date) -> Path:
    iso_year, iso_week, _ = day.isocalendar()
    return reports_dir / f"weekly_calibration_{iso_year}-W{iso_week:02d}.json"


def _plan_days(
    days: List[date],
    *,
    reports_dir: Path,
    force: bool,
) -> Tuple[List[date], List[date]]:
    target_days: List[date] = []
    skipped_days: List[date] = []
    for day in days:
        weekly_json = _weekly_json_path(reports_dir, day)
        if weekly_json.exists() and not force:
            skipped_days.append(day)
            continue
        target_days.append(day)
    return target_days, skipped_days


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-root", default="trace_v1")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    trace_root = Path(args.trace_root)
    reports_dir = Path(args.reports_dir)
    days = _discover_days(trace_root)
    if not days:
        print(f"[info] no trace day directories found: {trace_root}")
        return

    target_days, skipped_days = _plan_days(
        days,
        reports_dir=reports_dir,
        force=bool(args.force),
    )

    print(f"[info] discovered trace days: {len(days)}")
    print(f"[info] target days: {len(target_days)}")
    print(f"[info] skipped days (existing weekly): {len(skipped_days)}")
    if skipped_days:
        print(
            "[info] skipped list: "
            + ", ".join(day.isoformat() for day in skipped_days)
        )

    failed_days: List[date] = []
    for day in target_days:
        cmd = [
            sys.executable,
            "scripts/run_nightly_audit.py",
            "--trace_root",
            str(trace_root),
            "--day",
            day.isoformat(),
        ]
        if args.dry_run:
            print("[dry-run]", " ".join(cmd))
            continue
        print("[run]", " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"[warn] failed day={day.isoformat()} returncode={result.returncode}")
            failed_days.append(day)

    if failed_days:
        print(
            "[warn] failed dates: "
            + ", ".join(day.isoformat() for day in failed_days)
        )

    weekly_count = len(list(reports_dir.glob("weekly_calibration_*-W*.json")))
    print(f"[info] weekly calibration json count: {weekly_count}")


if __name__ == "__main__":
    main()
