#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eqnet.telemetry.change_decision_writer import (
    ChangeDecisionWriter,
    ChangeDecisionWriterConfig,
)


def _iso_week_string_from_timestamp_ms(timestamp_ms: int) -> str:
    dt = datetime.fromtimestamp(int(timestamp_ms) / 1000, tz=timezone.utc)
    year, week, _ = dt.isocalendar()
    return f"{year}-W{week:02d}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Append a manual change decision record.")
    ap.add_argument("--telemetry_dir", default="telemetry", help="Telemetry output directory.")
    ap.add_argument("--proposal_id", required=True, help="Target proposal_id.")
    ap.add_argument(
        "--decision",
        required=True,
        choices=[
            "REJECT",
            "ACCEPT_SHADOW",
            "ACCEPT_CANARY",
            "ACCEPT_ROLLOUT",
            "ROLLBACK",
        ],
        help="Decision type.",
    )
    ap.add_argument("--reason", required=True, help="Short reason for audit logs.")
    ap.add_argument("--actor", default="human", choices=["human", "auto"], help="Decision actor.")
    ap.add_argument("--timestamp_ms", type=int, default=0, help="Override timestamp in milliseconds.")
    args = ap.parse_args()

    timestamp_ms = int(args.timestamp_ms) if int(args.timestamp_ms) > 0 else int(datetime.now(timezone.utc).timestamp() * 1000)
    source_week = _iso_week_string_from_timestamp_ms(timestamp_ms)
    writer = ChangeDecisionWriter(
        ChangeDecisionWriterConfig(telemetry_dir=Path(args.telemetry_dir))
    )
    out_path = writer.append(
        timestamp_ms=timestamp_ms,
        proposal_id=args.proposal_id,
        decision=args.decision,
        actor=args.actor,
        reason=args.reason,
        source_week=source_week,
        decision_id=None,
        extra=None,
    )
    print(f"[info] change decision recorded: {out_path}")


if __name__ == "__main__":
    main()
