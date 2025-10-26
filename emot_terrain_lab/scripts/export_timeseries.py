# -*- coding: utf-8 -*-
"""
Export field metrics enriched with rest-mode signals and optional SVaR risk
columns for downstream analytics.

Usage:
    python scripts/export_timeseries.py --state data/state --out exports/timeseries.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from terrain.risk import (  # noqa: E402
    PercentileConfig,
    add_risk_flags,
    compute_percentiles,
    latest_snapshot,
)


def load_json(path: Path) -> Dict | List:
    return json.loads(path.read_text(encoding="utf-8"))


def build_rest_index(rest_payload: Dict) -> Dict[str, Dict]:
    history = rest_payload.get("history", [])
    index: Dict[str, Dict] = {}
    for entry in history:
        ts_raw = entry.get("timestamp")
        if not ts_raw:
            continue
        try:
            day = datetime.fromisoformat(ts_raw).date().isoformat()
        except Exception:
            continue
        index[day] = entry
    return index


def enrich_metrics(field_metrics: List[Dict], rest_index: Dict[str, Dict]) -> List[Dict]:
    enriched = []
    for record in field_metrics:
        ts_raw = record.get("timestamp")
        if not ts_raw:
            continue
        try:
            dt = datetime.fromisoformat(ts_raw)
        except Exception:
            continue
        day_key = dt.date().isoformat()
        rest_entry = rest_index.get(day_key, {})
        metrics = {
            "timestamp": ts_raw,
            "entropy": record.get("entropy"),
            "enthalpy_mean": record.get("enthalpy_mean"),
            "dissipation": record.get("dissipation"),
            "info_flux": record.get("info_flux"),
            "rest_flag": 1 if rest_entry else 0,
            "rest_reason": rest_entry.get("reason", ""),
            "rest_entropy": rest_entry.get("metrics", {}).get("entropy"),
            "rest_enthalpy": rest_entry.get("metrics", {}).get("enthalpy"),
            "rest_triggers_fatigue": rest_entry.get("triggers", {}).get("fatigue"),
            "rest_triggers_loop": rest_entry.get("triggers", {}).get("loop"),
            "rest_triggers_overload": rest_entry.get("triggers", {}).get("overload"),
        }
        enriched.append(metrics)
    return enriched


def write_csv(rows: List[Dict], out_path: Path) -> None:
    if not rows:
        print("No metrics to export.")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


def parse_percentiles(raw: str) -> List[float]:
    values: List[float] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(float(chunk))
        except ValueError as exc:
            raise ValueError(f"Invalid percentile value: {chunk}") from exc
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=str, required=True, help="Path to state directory, e.g., data/state")
    parser.add_argument("--out", type=str, required=True, help="Output CSV path, e.g., exports/timeseries.csv")
    parser.add_argument(
        "--risk-metrics",
        type=str,
        default="entropy,enthalpy_mean,dissipation,info_flux",
        help="Comma-separated metrics to include in SVaR percentile calculations",
    )
    parser.add_argument(
        "--risk-percentiles",
        type=str,
        default="90,95,99",
        help="Comma-separated percentile values (0-100) for SVaR summary, e.g., 90,95,99",
    )
    parser.add_argument(
        "--risk-high",
        type=float,
        default=95.0,
        help="Percentile label used for the per-row risk flag (default: 95)",
    )
    parser.add_argument(
        "--risk-summary",
        type=str,
        help="Optional JSON file to store percentile thresholds and latest snapshot",
    )
    parser.add_argument(
        "--disable-risk",
        action="store_true",
        help="Skip SVaR calculations (legacy output behaviour)",
    )
    args = parser.parse_args()

    state_dir = Path(args.state)
    metrics_path = state_dir / "field_metrics.json"
    rest_path = state_dir / "rest_state.json"

    field_metrics = load_json(metrics_path)
    rest_payload = load_json(rest_path)

    rest_index = build_rest_index(rest_payload)
    rows = enrich_metrics(field_metrics, rest_index)

    summary_path: Optional[Path] = None
    if args.risk_summary:
        summary_path = Path(args.risk_summary)

    if not args.disable_risk and rows:
        metric_keys = [key.strip() for key in args.risk_metrics.split(",") if key.strip()]
        percentile_values = parse_percentiles(args.risk_percentiles)

        config = PercentileConfig(metrics=metric_keys, percentiles=percentile_values)
        percentiles = compute_percentiles(field_metrics, config)
        high_label = f"p{int(max(0, min(100, args.risk_high)))}"
        rows = add_risk_flags(rows, percentiles, high_label)

        if summary_path:
            payload = {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "state_dir": state_dir.as_posix(),
                "high_watermark": high_label,
                "percentiles": percentiles,
                "latest": latest_snapshot(rows, percentiles),
            }
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote SVaR summary to {summary_path}")
    elif summary_path:
        print("Risk summary path provided but risk calculations disabled; skipping summary output.")

    write_csv(rows, Path(args.out))


if __name__ == "__main__":
    main()
