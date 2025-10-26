# -*- coding: utf-8 -*-
"""Nightly aggregation for Five-Sense-First telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator

from emot_terrain_lab.sense.sharedness import SharednessSample, summarise


def _iter_receipts(paths: Iterable[Path]) -> Iterator[dict]:
    for path in paths:
        if not path.exists():
            continue
        if path.is_dir():
            yield from _iter_receipts(path.glob("*.json"))
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            yield data


def aggregate(paths: Iterable[Path]) -> dict:
    samples = []
    for receipt in _iter_receipts(paths):
        sense = receipt.get("sense", {})
        residual = sense.get("residual", {})
        delta = float(residual.get("delta", 0.0))
        lang_loss = float(sense.get("language_loss", {}).get("loss", 0.0))
        samples.append(SharednessSample(delta=delta, language_loss=lang_loss))
    summary = summarise(samples)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receipts",
        type=Path,
        nargs="+",
        default=[Path("logs/receipts")],
        help="Paths to JSON receipts or directories.",
    )
    args = parser.parse_args()
    summary = aggregate(args.receipts)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
