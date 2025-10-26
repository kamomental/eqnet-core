# -*- coding: utf-8 -*-
"""Backfill helper for fast-path coverage/override baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


def load_fastpath_reports(report_dir: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not report_dir.exists():
        return entries
    for path in sorted(report_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        fast = data.get("fastpath") or {}
        if not fast:
            continue
        entries.append(
            {
                "path": str(path),
                "ts": data.get("ts"),
                "coverage_rate": fast.get("coverage_rate"),
                "override_rate": fast.get("override_rate"),
                "profiles": fast.get("profiles") or {},
            }
        )
    return entries


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    numbers = [float(v) for v in values if v is not None]
    if not numbers:
        return None
    return mean(numbers)


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[k]


def summarize_fastpath_metrics(
    entries: List[Dict[str, Any]], *, limit: Optional[int] = None
) -> Dict[str, Any]:
    if limit is not None and limit > 0:
        entries = entries[-limit:]
    coverage_rates = [entry.get("coverage_rate") for entry in entries]
    override_rates = [entry.get("override_rate") for entry in entries]
    summary: Dict[str, Any] = {
        "total_reports": len(entries),
        "coverage_rate_avg": _mean(coverage_rates),
        "override_rate_avg": _mean(override_rates),
        "override_rate_p95": _percentile(
            [float(v) for v in override_rates if v is not None], 95.0
        ),
        "profiles": {},
    }
    profile_window: Dict[str, Dict[str, List[float]]] = {}
    for entry in entries:
        for name, stats in (entry.get("profiles") or {}).items():
            window = profile_window.setdefault(
                name, {"coverage_rate": [], "override_rate": []}
            )
            cov = stats.get("coverage_rate")
            ov = stats.get("override_rate")
            if cov is not None:
                window["coverage_rate"].append(float(cov))
            if ov is not None:
                window["override_rate"].append(float(ov))
    for name, windows in profile_window.items():
        summary["profiles"][name] = {
            "coverage_rate_avg": _mean(windows["coverage_rate"]),
            "override_rate_avg": _mean(windows["override_rate"]),
        }
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports",
        type=Path,
        default=Path("reports/nightly"),
        help="Directory containing nightly JSON reports",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to persist the aggregated summary",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Limit aggregation to the most recent N reports (default: 30)",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    entries = load_fastpath_reports(args.reports)
    summary = summarize_fastpath_metrics(entries, limit=args.limit)
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    print(payload)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["load_fastpath_reports", "main", "summarize_fastpath_metrics"]
