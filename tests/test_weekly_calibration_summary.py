from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_weekly_calibration import _build_summary, _load_weekly_records


def _write_weekly(path: Path, *, week: str, action_code: str, changed_keys: list[str], sat: float, low: float) -> None:
    payload = {
        "schema_version": 1,
        "week": week,
        "day": week.replace("-W", "-") + "-1",
        "recommended_action_code": action_code,
        "changed_keys": changed_keys,
        "metrics": {
            "sat_p95_avg": sat,
            "low_ratio_avg": low,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_summary_tracks_none_streak_and_deltas(tmp_path: Path) -> None:
    _write_weekly(
        tmp_path / "weekly_calibration_2026-W01.json",
        week="2026-W01",
        action_code="none",
        changed_keys=[],
        sat=0.40,
        low=0.10,
    )
    _write_weekly(
        tmp_path / "weekly_calibration_2026-W02.json",
        week="2026-W02",
        action_code="none",
        changed_keys=[],
        sat=0.55,
        low=0.12,
    )
    _write_weekly(
        tmp_path / "weekly_calibration_2026-W03.json",
        week="2026-W03",
        action_code="saturation_high",
        changed_keys=[],
        sat=0.70,
        low=0.22,
    )
    rows = _load_weekly_records(tmp_path, max_weeks=12)
    summary = _build_summary(rows, sat_threshold=0.10, low_threshold=0.05)
    assert summary["none_streak"]["max"] == 2
    assert len(summary["delta_spike_weeks"]) >= 1
    spike = summary["delta_spike_weeks"][-1]
    assert "action_code" in spike
    assert "changed_keys_top" in spike


def test_build_summary_proposal_proxy_follow_and_match(tmp_path: Path) -> None:
    _write_weekly(
        tmp_path / "weekly_calibration_2026-W10.json",
        week="2026-W10",
        action_code="saturation_high",
        changed_keys=[],
        sat=0.80,
        low=0.20,
    )
    _write_weekly(
        tmp_path / "weekly_calibration_2026-W11.json",
        week="2026-W11",
        action_code="stable",
        changed_keys=["assoc_clamp.max", "assoc_weights.temporal"],
        sat=0.60,
        low=0.18,
    )
    rows = _load_weekly_records(tmp_path, max_weeks=12)
    summary = _build_summary(rows, sat_threshold=0.10, low_threshold=0.05)
    proxy = summary["proposal_proxy"]
    assert proxy["candidate_count"] == 1
    assert proxy["followed_count_proxy"] == 1
    assert proxy["match_count_proxy"] == 1


def test_load_weekly_records_skips_unsupported_schema(tmp_path: Path, caplog) -> None:
    bad = {
        "schema_version": 2,
        "week": "2026-W20",
        "recommended_action_code": "none",
        "changed_keys": [],
        "metrics": {"sat_p95_avg": 0.1, "low_ratio_avg": 0.1},
    }
    good = {
        "schema_version": 1,
        "week": "2026-W21",
        "recommended_action_code": "none",
        "changed_keys": [],
        "metrics": {"sat_p95_avg": 0.1, "low_ratio_avg": 0.1},
    }
    (tmp_path / "weekly_calibration_2026-W20.json").write_text(json.dumps(bad), encoding="utf-8")
    (tmp_path / "weekly_calibration_2026-W21.json").write_text(json.dumps(good), encoding="utf-8")
    caplog.set_level("WARNING")
    rows = _load_weekly_records(tmp_path, max_weeks=12)
    assert len(rows) == 1
    assert rows[0].week == "2026-W21"
    assert "unsupported schema_version=2" in caplog.text
