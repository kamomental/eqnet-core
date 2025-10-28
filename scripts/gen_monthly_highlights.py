#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from emot_terrain_lab.ops.monthly_highlights import generate_value_influence_highlights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate monthly value influence highlights.")
    parser.add_argument("--month", help="Target month in YYYY-MM (defaults to current UTC month).")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of top entries to keep.",
    )
    parser.add_argument(
        "--log-path",
        default=str(Path("logs/pain/value_influence.jsonl")),
        help="Path to value influence log.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("reports/monthly")),
        help="Directory to write highlight artifacts.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Compute highlights without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = generate_value_influence_highlights(
        month=args.month,
        limit=args.limit,
        log_path=args.log_path,
        output_dir=args.output_dir,
        write_files=not args.no_write,
    )
    items = payload.get("items", [])
    print(f"[monthly] month={payload.get('month')} items={len(items)}")
    if payload.get("paths"):
        print(f"[monthly] paths={payload['paths']}")


if __name__ == "__main__":
    main()
