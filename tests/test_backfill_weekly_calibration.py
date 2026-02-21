from __future__ import annotations

from datetime import date
from pathlib import Path

from scripts.backfill_weekly_calibration import _plan_days, _weekly_json_path


def test_weekly_json_path_uses_iso_week() -> None:
    path = _weekly_json_path(Path("reports"), date(2026, 1, 4))
    assert path.name == "weekly_calibration_2026-W01.json"


def test_plan_days_skips_existing_when_not_force(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    existing = reports / "weekly_calibration_2026-W01.json"
    existing.write_text("{}", encoding="utf-8")

    days = [date(2026, 1, 4), date(2026, 1, 19)]
    target, skipped = _plan_days(days, reports_dir=reports, force=False)
    assert target == [date(2026, 1, 19)]
    assert skipped == [date(2026, 1, 4)]


def test_plan_days_force_ignores_existing(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    existing = reports / "weekly_calibration_2026-W01.json"
    existing.write_text("{}", encoding="utf-8")

    days = [date(2026, 1, 4)]
    target, skipped = _plan_days(days, reports_dir=reports, force=True)
    assert target == [date(2026, 1, 4)]
    assert skipped == []

