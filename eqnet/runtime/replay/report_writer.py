from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def write_replay_reports(
    *,
    out_dir: Path,
    summary: Dict[str, Any],
    daily_rows: List[Dict[str, Any]],
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "replay_summary.json"
    daily_path = out_dir / "replay_daily.jsonl"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [json.dumps(row, ensure_ascii=False) for row in daily_rows]
    daily_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {"summary": summary_path, "daily": daily_path}


def write_diff_reports(
    *,
    out_dir: Path,
    diff_summary: Dict[str, Any],
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    diff_json = out_dir / "diff_summary.json"
    diff_md = out_dir / "diff_top_changes.md"
    diff_json.write_text(json.dumps(diff_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Replay Diff Top Changes", ""]
    for row in diff_summary.get("top_changes") or []:
        metric = str(row.get("metric") or "unknown")
        delta = float(row.get("delta") or 0.0)
        lines.append(f"- {metric}: {delta:+.6f}")
    diff_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"diff_summary": diff_json, "diff_md": diff_md}

