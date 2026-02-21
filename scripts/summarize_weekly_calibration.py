#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize weekly calibration logs into operational signals."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_LOGGER = logging.getLogger(__name__)
SUPPORTED_SCHEMA_VERSIONS = {1}


@dataclass
class WeeklyRecord:
    path: Path
    week: str
    day: str
    action_code: str
    action_text: str
    changed_keys: List[str]
    metrics: Dict[str, Any]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _week_key(week: str) -> Tuple[int, int]:
    # week format: YYYY-WW
    try:
        year_s, week_s = week.split("-W", 1)
        return int(year_s), int(week_s)
    except Exception:
        return (0, 0)


def _load_weekly_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    raw = payload.get("schema_version", 1)
    try:
        schema_version = int(raw)
    except (TypeError, ValueError):
        schema_version = -1
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        _LOGGER.warning(
            "summarize_weekly_calibration: unsupported schema_version=%s, skip file=%s",
            raw,
            path,
        )
        return None
    return payload


def _load_weekly_records(weekly_dir: Path, *, max_weeks: int) -> List[WeeklyRecord]:
    rows: List[WeeklyRecord] = []
    for path in sorted(weekly_dir.glob("weekly_calibration_*-W*.json")):
        payload = _load_weekly_json(path)
        if payload is None:
            continue
        week = str(payload.get("week", "")).strip()
        if not week:
            continue
        rows.append(
            WeeklyRecord(
                path=path,
                week=week,
                day=str(payload.get("day", "")),
                action_code=str(payload.get("recommended_action_code", "none") or "none"),
                action_text=str(payload.get("recommended_action", "") or ""),
                changed_keys=list(payload.get("changed_keys", []) or []),
                metrics=dict(payload.get("metrics", {}) or {}),
            )
        )
    rows.sort(key=lambda r: _week_key(r.week))
    if max_weeks > 0 and len(rows) > max_weeks:
        rows = rows[-max_weeks:]
    return rows


def _none_streak_stats(records: List[WeeklyRecord]) -> Dict[str, int]:
    max_streak = 0
    cur = 0
    for row in records:
        if row.action_code == "none":
            cur += 1
            if cur > max_streak:
                max_streak = cur
        else:
            cur = 0
    return {"current": cur, "max": max_streak}


def _delta_spike_weeks(
    records: List[WeeklyRecord],
    *,
    sat_threshold: float,
    low_threshold: float,
    max_changed_keys: int = 3,
) -> List[Dict[str, Any]]:
    spikes: List[Dict[str, Any]] = []
    prev_sat: Optional[float] = None
    prev_low: Optional[float] = None
    for row in records:
        sat = _safe_float(row.metrics.get("sat_p95_avg"), 0.0)
        low = _safe_float(row.metrics.get("low_ratio_avg"), 0.0)
        if prev_sat is not None and prev_low is not None:
            ds = sat - prev_sat
            dl = low - prev_low
            if abs(ds) >= sat_threshold or abs(dl) >= low_threshold:
                spikes.append(
                    {
                        "week": row.week,
                        "delta_sat_p95_avg": round(ds, 3),
                        "delta_low_ratio_avg": round(dl, 3),
                        "action_code": row.action_code,
                        "recommended_action": row.action_text,
                        "changed_keys_top": row.changed_keys[: max(1, max_changed_keys)],
                    }
                )
        prev_sat = sat
        prev_low = low
    return spikes


def _is_followed(action_code: str, next_changed_keys: List[str]) -> bool:
    if action_code == "saturation_high":
        return any(
            key.startswith("assoc_clamp.")
            or key.startswith("assoc_weights.")
            or key == "assoc_temporal_tau_sec"
            for key in next_changed_keys
        )
    if action_code == "retrieval_sparse":
        return any(
            key.startswith("assoc_weights.")
            or key == "assoc_temporal_tau_sec"
            for key in next_changed_keys
        )
    if action_code == "uncertainty_high":
        return any(
            key in {"confidence_low_max", "confidence_mid_max"}
            or key.startswith("assoc_")
            for key in next_changed_keys
        )
    return False


def _is_improved(action_code: str, cur: WeeklyRecord, nxt: WeeklyRecord) -> bool:
    cur_sat = _safe_float(cur.metrics.get("sat_p95_avg"), 0.0)
    next_sat = _safe_float(nxt.metrics.get("sat_p95_avg"), 0.0)
    cur_low = _safe_float(cur.metrics.get("low_ratio_avg"), 0.0)
    next_low = _safe_float(nxt.metrics.get("low_ratio_avg"), 0.0)
    if action_code == "saturation_high":
        return next_sat <= cur_sat
    if action_code == "retrieval_sparse":
        return next_low <= cur_low
    if action_code == "uncertainty_high":
        return (next_low <= cur_low) and (next_sat <= cur_sat + 0.02)
    return True


def _proposal_proxy_stats(records: List[WeeklyRecord]) -> Dict[str, Any]:
    # proxy:
    # - followed: next week changed_keys include expected keys for current action_code
    # - matched: followed and next week metrics moved in expected direction
    candidates = [i for i, r in enumerate(records[:-1]) if r.action_code not in {"none", "stable"}]
    followed = 0
    matched = 0
    details: List[Dict[str, Any]] = []
    for idx in candidates:
        cur = records[idx]
        nxt = records[idx + 1]
        is_followed = _is_followed(cur.action_code, nxt.changed_keys)
        if is_followed:
            followed += 1
        is_matched = is_followed and _is_improved(cur.action_code, cur, nxt)
        if is_matched:
            matched += 1
        details.append(
            {
                "week": cur.week,
                "action_code": cur.action_code,
                "next_week": nxt.week,
                "followed_proxy": is_followed,
                "outcome_match_proxy": is_matched,
            }
        )
    total = len(candidates)
    follow_rate = (followed / total) if total else 0.0
    match_rate = (matched / followed) if followed else 0.0
    return {
        "candidate_count": total,
        "followed_count_proxy": followed,
        "follow_rate_proxy": round(follow_rate, 3),
        "match_count_proxy": matched,
        "match_rate_proxy": round(match_rate, 3),
        "details": details,
    }


def _build_summary(
    records: List[WeeklyRecord],
    *,
    sat_threshold: float,
    low_threshold: float,
) -> Dict[str, Any]:
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "schema_version": 1,
        "weeks": [r.week for r in records],
        "none_streak": _none_streak_stats(records),
        "delta_spike_weeks": _delta_spike_weeks(
            records,
            sat_threshold=sat_threshold,
            low_threshold=low_threshold,
        ),
        "proxy_definition": {
            "follow_rate_proxy": "前週のaction_codeに対応する設定キーが当週changed_keysに含まれる割合（近似）",
            "match_rate_proxy": "推奨方向と次週メトリクス変化が一致した割合（近似）",
            "note": "proxyはログ構造に基づく保守的推定であり、厳密な採用/一致判定ではない",
        },
        "proposal_proxy": _proposal_proxy_stats(records),
    }


def _to_markdown(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Weekly Calibration Ops Summary")
    lines.append("")
    lines.append(f"- generated_at: {summary.get('generated_at')}")
    lines.append(f"- schema_version: {summary.get('schema_version')}")
    lines.append(f"- weeks: {summary.get('weeks', [])}")
    lines.append("")
    none_streak = summary.get("none_streak", {})
    lines.append("## None Streak")
    lines.append(f"- current: {none_streak.get('current', 0)}")
    lines.append(f"- max: {none_streak.get('max', 0)}")
    lines.append("")
    lines.append("## Delta Spike Weeks")
    spikes = summary.get("delta_spike_weeks", [])
    if spikes:
        for s in spikes:
            changed = s.get("changed_keys_top", [])
            changed_text = ", ".join(changed) if changed else "none"
            lines.append(
                f"- {s.get('week')}: Δsat_avg={s.get('delta_sat_p95_avg'):+.3f}, "
                f"Δlow_avg={s.get('delta_low_ratio_avg'):+.3f}, "
                f"action={s.get('action_code', 'none')}, changed={changed_text}"
            )
    else:
        lines.append("- none")
    lines.append("")
    proposal = summary.get("proposal_proxy", {})
    lines.append("## Proposal Proxy")
    lines.append(f"- candidate_count: {proposal.get('candidate_count', 0)}")
    lines.append(
        f"- follow_rate_proxy: {proposal.get('follow_rate_proxy', 0.0)} "
        f"({proposal.get('followed_count_proxy', 0)}/{proposal.get('candidate_count', 0)})"
    )
    lines.append(
        f"- match_rate_proxy: {proposal.get('match_rate_proxy', 0.0)} "
        f"({proposal.get('match_count_proxy', 0)}/{proposal.get('followed_count_proxy', 0)})"
    )
    lines.append("")
    lines.append("## Proxy Definition")
    proxy_def = summary.get("proxy_definition", {})
    lines.append(f"- follow_rate_proxy: {proxy_def.get('follow_rate_proxy', '')}")
    lines.append(f"- match_rate_proxy: {proxy_def.get('match_rate_proxy', '')}")
    lines.append(f"- note: {proxy_def.get('note', '')}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weekly-dir", default="reports", help="Directory containing weekly_calibration_*.json")
    ap.add_argument("--weeks", type=int, default=12, help="Number of recent weeks to include (0=all)")
    ap.add_argument("--delta-sat-threshold", type=float, default=0.10)
    ap.add_argument("--delta-low-threshold", type=float, default=0.05)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-md", default="")
    args = ap.parse_args()

    weekly_dir = Path(args.weekly_dir)
    records = _load_weekly_records(weekly_dir, max_weeks=int(args.weeks))
    summary = _build_summary(
        records,
        sat_threshold=float(args.delta_sat_threshold),
        low_threshold=float(args.delta_low_threshold),
    )

    stamp = datetime.utcnow().strftime("%Y%m%d")
    out_json = Path(args.out_json) if args.out_json else weekly_dir / f"weekly_ops_summary_{stamp}.json"
    out_md = Path(args.out_md) if args.out_md else weekly_dir / f"weekly_ops_summary_{stamp}.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_to_markdown(summary), encoding="utf-8")
    print(f"[info] weekly ops summary json: {out_json}")
    print(f"[info] weekly ops summary md: {out_md}")


if __name__ == "__main__":
    main()

