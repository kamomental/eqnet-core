#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert trace_v1 jsonl into a replay.json for the RPG viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DECISION_MAP = {
    "execute": "PASS",
    "cancel": "VETO",
    "hold": "HOLD",
}


def _decision_from_trace(record: Dict[str, Any]) -> str:
    event_type = record.get("event_type")
    if event_type == "world_transition":
        return "HOLD"
    accepted = (record.get("prospection") or {}).get("accepted")
    if accepted is True:
        return "PASS"
    if accepted is False:
        return "VETO"
    return "UNKNOWN"


def _collect_trace_files(trace_dir: Path) -> List[Path]:
    if trace_dir.is_file():
        return [trace_dir]
    if not trace_dir.exists():
        raise FileNotFoundError(f"trace_dir not found: {trace_dir}")
    return sorted(trace_dir.glob("*.jsonl"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", required=True, help="Path to trace_v1/YYYY-MM-DD or a single jsonl")
    ap.add_argument("--out", dest="out_file", default="replay.json")
    args = ap.parse_args()

    trace_dir = Path(args.trace_dir)
    events: List[Dict[str, Any]] = []
    for fp in _collect_trace_files(trace_dir):
        with fp.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                decision = _decision_from_trace(record)
                boundary = record.get("boundary") or {}
                reasons = boundary.get("reasons") or {}
                events.append(
                    {
                        "ts": record.get("timestamp_ms"),
                        "seed": record.get("seed"),
                        "world_type": record.get("world_type") or record.get("scenario_id"),
                        "decision": decision,
                        "risk_pre": reasons.get("risk"),
                        "risk_post": reasons.get("risk"),
                        "post_risk_scale": record.get("post_risk_scale"),
                        "uncertainty_factor": record.get("transition_uncertainty_factor"),
                        "decision_reason": record.get("decision_reason"),
                        "world_snapshot": record.get("world_snapshot"),
                        "override_context": record.get("override_context"),
                        "postmortem_note": record.get("postmortem_note"),
                        "_file": fp.name,
                        "_line": line_no,
                        "_event_type": record.get("event_type"),
                    }
                )

    def sort_key(ev: Dict[str, Any]) -> tuple[bool, Any]:
        ts = ev.get("ts")
        return (ts is None, ts)

    events.sort(key=sort_key)
    out_path = Path(args.out_file)
    out_path.write_text(json.dumps({"events": events}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_path} with {len(events)} events")


if __name__ == "__main__":
    main()
