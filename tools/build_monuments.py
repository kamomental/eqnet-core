# -*- coding: utf-8 -*-
"""Detect Monument events from Self-Report + Narrative logs (meaning-based)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def _fast_triggers(entry: Dict[str, Any]) -> Tuple[bool, List[str]]:
    metrics = entry.get("metrics") or {}
    love = float(metrics.get("love", 0.0) or 0.0)
    override = float(metrics.get("fastpath_override", 0.0) or 0.0)
    tension = float(metrics.get("tension", 0.0) or 0.0)
    mood = str(entry.get("mood") or "").lower()
    tone = str(entry.get("social_tone") or "").lower()

    reasons: List[str] = []
    if override >= 0.3:
        reasons.append("high_override")
    if mood == "warm" and love < 0.18:
        reasons.append("warm_despite_low_love")
    if mood == "guarded" and love > 0.22:
        reasons.append("guarded_under_high_love")
    if tone == "supportive" and tension > 0.25:
        reasons.append("supportive_under_tension")
    if not reasons and abs(tension) < 0.05 and override > 0.0 and mood in {"calm", "steady"}:
        reasons.append("quiet_override")
    return (bool(reasons), reasons)


def _narrative_map(df_narr: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    if df_narr.empty:
        return mapping
    for _, row in df_narr.iterrows():
        episode_ids = row.get("episode_ids") or []
        if not isinstance(episode_ids, list):
            continue
        for ep in episode_ids:
            mapping[str(ep)] = row.to_dict()
    return mapping


def _slow_triggers(
    narrative: Optional[Dict[str, Any]],
    previous_theme: Optional[str],
) -> Tuple[bool, List[str]]:
    if not narrative:
        return False, []
    reasons: List[str] = []
    theme = str(narrative.get("theme") or "")
    emotional = str(narrative.get("emotional_trend") or "").lower()
    description = str(narrative.get("description") or "").lower()

    if previous_theme and theme and theme != previous_theme:
        reasons.append(f"theme_shift:{previous_theme}->{theme}")
    if emotional in {"warm", "transition", "breakthrough"}:
        reasons.append(f"emotional_{emotional}")
    if "special" in description or "memorable" in description:
        reasons.append("narrative_marked_special")
    return bool(reasons), reasons


def build_monuments(
    df_self: pd.DataFrame,
    df_narr: pd.DataFrame,
) -> List[Dict[str, Any]]:
    if df_self.empty:
        return []
    df_self = df_self.sort_values(by="ts")
    narrative_map = _narrative_map(df_narr)
    prev_theme_by_tag: Dict[str, Optional[str]] = {}
    monuments: List[Dict[str, Any]] = []

    for _, entry in df_self.iterrows():
        entry_dict = entry.to_dict()
        fast, fast_reasons = _fast_triggers(entry_dict)
        if not fast:
            continue
        episode_id = str(entry_dict.get("episode_id"))
        tag = str(entry_dict.get("tag")) if entry_dict.get("tag") else "default"
        narrative = narrative_map.get(episode_id)
        prev_theme = prev_theme_by_tag.get(tag)
        slow, slow_reasons = _slow_triggers(narrative, prev_theme)
        if narrative and narrative.get("theme"):
            prev_theme_by_tag[tag] = narrative.get("theme")
        if not slow:
            continue
        monuments.append(
            {
                "ts": entry_dict.get("ts"),
                "episode_id": episode_id,
                "tag": tag,
                "mood": entry_dict.get("mood"),
                "summary": entry_dict.get("summary"),
                "fast_reasons": fast_reasons,
                "slow_reasons": slow_reasons,
                "narrative_id": narrative.get("narrative_id") if narrative else None,
                "narrative_theme": narrative.get("theme") if narrative else None,
                "metrics": entry_dict.get("metrics"),
            }
        )
    return monuments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-report", type=Path, default=Path("logs/self_report.jsonl"))
    parser.add_argument("--narrative", type=Path, default=Path("logs/narrative_log.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("logs/monuments.jsonl"))
    args = parser.parse_args()

    df_self = read_jsonl(args.self_report)
    df_narr = read_jsonl(args.narrative)
    monuments = build_monuments(df_self, df_narr)
    if not monuments:
        print("[monuments] no monuments detected")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in monuments:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[monuments] wrote {len(monuments)} entries to {args.out}")


if __name__ == "__main__":
    main()


