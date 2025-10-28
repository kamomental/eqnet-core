from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from emot_terrain_lab.ops.monthly_highlights import (
    compute_value_influence_items,
    generate_value_influence_highlights,
)


def _ts(year: int, month: int, day: int) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp() * 1000)


def test_compute_value_influence_items():
    records = [
        {
            "ts_ms": _ts(2025, 1, 3),
            "harmed_values": ["belonging"],
            "care_target": True,
            "value_shift": {"delta_empathy_gain": 0.1},
        },
        {
            "ts_ms": _ts(2025, 1, 5),
            "harmed_values": ["belonging", "dignity"],
            "care_target": False,
            "value_shift": {"delta_empathy_gain": 0.2},
        },
        {
            "ts_ms": _ts(2025, 2, 1),
            "harmed_values": ["belonging"],
            "care_target": True,
            "value_shift": {"delta_empathy_gain": 0.5},
        },
    ]
    items = compute_value_influence_items(records, month="2025-01", limit=5)
    assert len(items) == 3
    assert items[0].value == "belonging"
    assert items[0].care_target is False
    assert abs(items[0].sum_delta_empathy_gain - 0.2) < 1e-6
    assert items[0].count == 1


def test_generate_value_influence_highlights(tmp_path):
    log_path = tmp_path / "value_influence.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sample = {
        "ts_ms": _ts(2025, 1, 10),
        "harmed_values": ["belonging"],
        "care_target": True,
        "value_shift": {"delta_empathy_gain": 0.25},
    }
    log_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    result = generate_value_influence_highlights(
        month="2025-01",
        limit=3,
        log_path=log_path,
        output_dir=tmp_path / "out",
        write_files=True,
    )
    paths = result.get("paths") or {}
    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).exists()
    assert result["items"][0]["value"] == "belonging"
