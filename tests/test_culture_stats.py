import json
from pathlib import Path

import pytest

from ops import nightly


def test_summarize_culture_stats(tmp_path: Path) -> None:
    log_path = tmp_path / "affective_log.jsonl"
    rows = [
        {
            "culture_tag": "JP_basic",
            "valence": 0.2,
            "arousal": 0.4,
            "rho": 0.5,
            "politeness": 0.7,
            "intimacy": 0.3,
        },
        {
            "culture_tag": "JP_basic",
            "valence": 0.4,
            "arousal": 0.6,
            "rho": 0.7,
            "politeness": 0.9,
        },
        {
            "culture_tag": "US_casual",
            "valence": -0.1,
            "arousal": 0.3,
            "rho": 0.25,
            "intimacy": 0.6,
        },
    ]
    log_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    stats = nightly._summarize_culture_stats(log_path)
    assert stats is not None
    assert "JP_basic" in stats
    assert "US_casual" in stats

    jp = stats["JP_basic"]
    assert pytest.approx(jp["count"], rel=1e-6) == 2.0
    assert pytest.approx(jp["mean_valence"], rel=1e-6) == 0.3
    assert pytest.approx(jp["mean_arousal"], rel=1e-6) == 0.5
    assert pytest.approx(jp["mean_rho"], rel=1e-6) == 0.6
    assert pytest.approx(jp["mean_politeness"], rel=1e-6) == 0.8
    assert pytest.approx(jp["mean_intimacy"], rel=1e-6) == 0.3

    us = stats["US_casual"]
    assert pytest.approx(us["count"], rel=1e-6) == 1.0
    assert pytest.approx(us["mean_valence"], rel=1e-6) == -0.1
    assert pytest.approx(us["mean_arousal"], rel=1e-6) == 0.3
    assert pytest.approx(us["mean_rho"], rel=1e-6) == 0.25
    assert pytest.approx(us["mean_intimacy"], rel=1e-6) == 0.6
    assert "mean_politeness" not in us
