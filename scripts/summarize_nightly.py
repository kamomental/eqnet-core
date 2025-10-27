#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarise Nightly outputs for pull-request comments.

Extracts alerts_detail and cultural statistics from reports/nightly.json
and prints a short Japanese memo when notable events occurred.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _format_alert(alert: Dict[str, Any]) -> str:
    kind = alert.get("kind", "unknown")
    tag = alert.get("tag")
    value = alert.get("value")
    threshold = alert.get("threshold")
    pair = alert.get("pair")
    parts = [f"- {kind}"]
    if tag:
        parts.append(f"(tag: {tag})")
    if pair:
        parts.append(f"(pair: {pair})")
    if value is not None and threshold is not None:
        parts.append(f": 値 {float(value):.3f} / 閾値 {float(threshold):.3f}")
    elif value is not None:
        parts.append(f": 値 {float(value):.3f}")
    return " ".join(parts)


def _format_culture(stats: Dict[str, Dict[str, Any]], top_k: int) -> List[str]:
    rows: List[tuple[float, str, Dict[str, Any]]] = []
    for tag, entry in stats.items():
        try:
            val = float(entry.get("mean_valence", 0.0))
        except (TypeError, ValueError):
            val = 0.0
        rows.append((abs(val), tag, entry))
    rows.sort(reverse=True)

    memo: List[str] = []
    for _, tag, entry in rows[:top_k]:
        val = float(entry.get("mean_valence", 0.0))
        rho = float(entry.get("mean_rho", 0.0))
        count = entry.get("count", 0)
        tendency = "ポジ寄り" if val >= 0 else "ネガ寄り"
        memo.append(f"- {tag}: {tendency} (valence {val:+.2f}) / ρ={rho:.2f} / n={count}")
    return memo


def _policy_delta(policy_feedback: Dict[str, Any]) -> Tuple[float | None, float | None, float | None]:
    try:
        before = float(policy_feedback.get("politeness_before"))
        after = float(policy_feedback.get("politeness_after"))
    except (TypeError, ValueError):
        return None, None, None
    return before, after, after - before


def should_emit_summary(report: Dict[str, Any], delta_threshold: float) -> bool:
    alerts_detail = report.get("alerts_detail") or []
    if alerts_detail:
        return True

    policy_feedback = report.get("policy_feedback")
    if isinstance(policy_feedback, dict) and policy_feedback.get("enabled"):
        _, _, delta = _policy_delta(policy_feedback)
        if delta is not None and abs(delta) >= delta_threshold:
            return True
    return False


def generate_summary(report: Dict[str, Any], top_k: int) -> str:
    lines: List[str] = ["### Nightly サマリ"]

    alerts_detail = report.get("alerts_detail") or []
    if alerts_detail:
        lines.append("**alerts**")
        for alert in alerts_detail:
            lines.append(_format_alert(alert))
    else:
        alerts = report.get("alerts") or []
        if alerts:
            lines.append("**alerts**")
            for raw in alerts:
                lines.append(f"- {raw}")

    culture_stats = report.get("culture_stats") or {}
    if culture_stats:
        lines.append("")
        lines.append("**文化トップ観測**")
        lines.extend(_format_culture(culture_stats, top_k))

    policy_feedback = report.get("policy_feedback")
    if isinstance(policy_feedback, dict) and policy_feedback.get("enabled"):
        before, after, delta = _policy_delta(policy_feedback)
        reason = policy_feedback.get("reason", "n/a")
        lines.append("")
        if before is not None and after is not None and delta is not None:
            lines.append(
                f"**ポリシーフィードバック** politeness {before:.3f} -> {after:.3f} "
                f"(delta={delta:+.3f}, reason={reason})"
            )
        else:
            lines.append(f"**ポリシーフィードバック** reason={reason}")

    return "\n".join(lines).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarise nightly.json for PR comments.")
    parser.add_argument("--json", default="reports/nightly.json", help="Nightly JSON path")
    parser.add_argument("--out", default="", help="Optional output file path")
    parser.add_argument("--top-k", type=int, default=3, help="Top culture tags to list")
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=0.01,
        help="Minimum |Δpoliteness| to report when no alerts fire.",
    )
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        print(f"[summary] nightly json not found: {json_path}")
        return 1

    report = json.loads(json_path.read_text(encoding="utf-8"))
    if not should_emit_summary(report, args.delta_threshold):
        if args.out:
            out_path = Path(args.out)
            if out_path.exists():
                out_path.unlink()
        else:
            print("[summary] no significant nightly events; skipping output")
        return 0

    body = generate_summary(report, args.top_k)
    if args.out:
        Path(args.out).write_text(body, encoding="utf-8")
    else:
        print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
