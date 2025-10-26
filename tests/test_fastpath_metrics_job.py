# -*- coding: utf-8 -*-

import json
from pathlib import Path

from ops.jobs.fastpath_metrics import (
    load_fastpath_reports,
    summarize_fastpath_metrics,
)


def _write_report(path: Path, coverage: float, override: float) -> None:
    report = {
        "ts": coverage * 10,
        "fastpath": {
            "coverage_rate": coverage,
            "override_rate": override,
            "profiles": {
                "cleanup": {
                    "coverage_rate": coverage,
                    "override_rate": override / 2.0,
                }
            },
        },
    }
    path.write_text(json.dumps(report), encoding="utf-8")


def test_fastpath_metrics_summary(tmp_path) -> None:
    _write_report(tmp_path / "20250101.json", 0.6, 0.1)
    _write_report(tmp_path / "20250102.json", 0.4, 0.05)

    entries = load_fastpath_reports(tmp_path)
    assert len(entries) == 2

    summary = summarize_fastpath_metrics(entries, limit=2)
    assert summary["total_reports"] == 2
    assert summary["override_rate_avg"] == 0.075
    profile_summary = summary["profiles"]["cleanup"]
    assert profile_summary["coverage_rate_avg"] == 0.5
    assert profile_summary["override_rate_avg"] == 0.0375
