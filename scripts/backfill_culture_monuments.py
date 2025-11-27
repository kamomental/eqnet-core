"""One-shot monument backfill from existing MomentLog entries."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from eqnet.culture_model import (
    MemoryMonument,
    MonumentStorage,
    rescan_events_for_monuments,
    use_monument_storage,
)
from eqnet.logs.moment_log import MomentLogEntry, iter_moment_entries


def _build_event(entry: MomentLogEntry, recurrence: int, text: str) -> Dict[str, object]:
    mood = entry.mood or {}
    metrics = entry.metrics or {}
    return {
        "ts": entry.ts,
        "culture_tag": entry.culture_tag,
        "place_id": getattr(entry, "place_id", None),
        "partner_id": getattr(entry, "partner_id", None),
        "object_id": getattr(entry, "object_id", None),
        "object_role": getattr(entry, "object_role", None),
        "activity_tag": getattr(entry, "activity_tag", None),
        "valence": mood.get("valence"),
        "arousal": mood.get("arousal"),
        "rho": metrics.get("rho") or metrics.get("rho_norm") or metrics.get("R"),
        "intimacy": metrics.get("intimacy"),
        "politeness": metrics.get("politeness"),
        "text": text,
        "recurrence": recurrence,
    }


def _iter_event_payloads(entries: Iterable[MomentLogEntry]) -> List[Dict[str, object]]:
    counters: Dict[Tuple[str, str, str, str], int] = defaultdict(int)
    payloads: List[Dict[str, object]] = []
    for entry in entries:
        key = (
            entry.culture_tag or "default",
            getattr(entry, "place_id", None) or "unknown_place",
            getattr(entry, "partner_id", None) or "solo",
            getattr(entry, "object_id", None) or "",
        )
        counters[key] += 1
        text_bits = [chunk for chunk in (entry.user_text, entry.llm_text) if chunk]
        text = " ".join(text_bits)
        payloads.append(_build_event(entry, counters[key], text))
    return payloads


def _write_output(path: Path, monuments: Iterable[MemoryMonument]) -> None:
    data = [asdict(mon) for mon in monuments]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill culture monuments from a MomentLog JSONL")
    parser.add_argument("moment_log", type=Path, help="Path to moment_log.jsonl")
    parser.add_argument("output", type=Path, help="Where to write the resulting monuments JSON")
    parser.add_argument("--reset", action="store_true", help="Clear in-memory monuments before rescanning")
    args = parser.parse_args()

    entries = list(iter_moment_entries(args.moment_log))
    if not entries:
        raise SystemExit(f"No entries found in {args.moment_log}")
    events = _iter_event_payloads(entries)

    storage = MonumentStorage()
    use_monument_storage(storage)
    created = rescan_events_for_monuments(
        events,
        text_getter=lambda event: str(event.get("text", "")),
        recurrence_getter=lambda event: int(event.get("recurrence", 0)),
        storage=storage,
        reset_existing=args.reset,
    )
    _write_output(args.output, storage.values())
    print(f"Processed {len(events)} events; monuments now: {len(storage)} (created/updated: {created})")


if __name__ == "__main__":
    main()
