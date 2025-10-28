from __future__ import annotations

import json
from calendar import monthrange
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from emot_terrain_lab.ops.pain_loop import VALUE_INFLUENCE_LOG
from emot_terrain_lab.utils.jsonl_io import read_jsonl


@dataclass
class InfluenceItem:
    value: str
    care_target: bool
    sum_delta_empathy_gain: float
    count: int

    @property
    def avg_delta_empathy_gain(self) -> float:
        return self.sum_delta_empathy_gain / self.count if self.count else 0.0


def _month_bounds(month: str) -> tuple[int, int]:
    """Return inclusive start and exclusive end timestamps (ms) for a month."""
    dt = datetime.strptime(month, "%Y-%m")
    start = datetime(dt.year, dt.month, 1, tzinfo=timezone.utc)
    last_day = monthrange(dt.year, dt.month)[1]
    end = datetime(dt.year, dt.month, last_day, 23, 59, 59, 999000, tzinfo=timezone.utc)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def _iter_influence_records(path: Path) -> Iterable[Dict]:
    if not path.exists():
        return []
    return read_jsonl(str(path))


def compute_value_influence_items(
    records: Iterable[Dict],
    *,
    month: str | None,
    limit: int = 5,
) -> List[InfluenceItem]:
    bounds = None
    if month:
        bounds = _month_bounds(month)
    aggregates: Dict[tuple[str, bool], Dict[str, float]] = {}
    for raw in records:
        try:
            ts_ms = int(raw.get("ts_ms"))
        except (TypeError, ValueError):
            continue
        if bounds:
            start, end = bounds
            if ts_ms < start or ts_ms > end:
                continue
        harmed_values: Sequence[str] = raw.get("harmed_values") or []
        if not harmed_values:
            continue
        value_shift = raw.get("value_shift") or {}
        try:
            delta_empathy = float(value_shift.get("delta_empathy_gain", 0.0))
        except (TypeError, ValueError):
            continue
        care_target = bool(raw.get("care_target"))
        for value in harmed_values:
            key = (str(value), care_target)
            bucket = aggregates.setdefault(
                key,
                {"sum": 0.0, "count": 0},
            )
            bucket["sum"] += delta_empathy
            bucket["count"] += 1
    items: List[InfluenceItem] = []
    for (value, care_flag), stats in aggregates.items():
        items.append(
            InfluenceItem(
                value=value,
                care_target=care_flag,
                sum_delta_empathy_gain=stats["sum"],
                count=int(stats["count"]),
            )
        )
    items.sort(key=lambda item: item.sum_delta_empathy_gain, reverse=True)
    if limit is not None and limit > 0:
        items = items[:limit]
    return items


def generate_value_influence_highlights(
    *,
    month: str | None = None,
    limit: int = 5,
    log_path: str | Path = VALUE_INFLUENCE_LOG,
    output_dir: str | Path | None = "reports/monthly",
    write_files: bool = True,
) -> Dict[str, object]:
    month = month or datetime.utcnow().strftime("%Y-%m")
    path = Path(log_path)
    records = _iter_influence_records(path)
    items = compute_value_influence_items(records, month=month, limit=limit)
    payload = {
        "month": month,
        "items": [
            {
                "value": item.value,
                "care_target": item.care_target,
                "sum_delta_empathy_gain": item.sum_delta_empathy_gain,
                "avg_delta_empathy_gain": item.avg_delta_empathy_gain,
                "count": item.count,
            }
            for item in items
        ],
    }
    paths: Dict[str, str | None] = {"json": None, "markdown": None}
    if write_files and output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"value_influence_top_{month}.json"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        paths["json"] = str(json_path)
        md_lines = [f"# Value Influence Top ({month})"]
        if items:
            for idx, item in enumerate(items, start=1):
                target_label = "care" if item.care_target else "all"
                md_lines.append(
                    f"{idx}. {item.value} / {target_label}: "
                    f"+{item.sum_delta_empathy_gain:.4f} (n={item.count}, avg={item.avg_delta_empathy_gain:.4f})"
                )
        else:
            md_lines.append("No value influence records for the selected month.")
        md_path = out_dir / f"value_influence_top_{month}.md"
        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        paths["markdown"] = str(md_path)
    payload["paths"] = paths
    return payload


__all__ = [
    "InfluenceItem",
    "compute_value_influence_items",
    "generate_value_influence_highlights",
]
