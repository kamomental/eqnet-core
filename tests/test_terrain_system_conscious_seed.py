import json
from pathlib import Path

from emot_terrain_lab.terrain.system import EmotionalMemorySystem


def test_latest_nightly_working_memory_replay_summary_merges_conscious_seed(tmp_path, monkeypatch) -> None:
    workspace = Path.cwd()
    reports_dir = workspace / "reports"
    reports_dir.mkdir(exist_ok=True)
    nightly_path = reports_dir / "nightly.json"
    backup = nightly_path.read_text(encoding="utf-8") if nightly_path.exists() else None
    nightly_path.write_text(
        json.dumps(
            {
                "inner_os_working_memory_replay_bias": {
                    "focus": "market",
                    "anchor": "market street",
                    "strength": 0.5,
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    conscious_path = tmp_path / "conscious_episodes.jsonl"
    conscious_path.write_text(
        json.dumps(
            {
                "id": "ep-1",
                "working_memory_seed": {
                    "focus": "harbor_slope",
                    "anchor": "harbor_slope",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CONSCIOUS_MEMORY_PATH", str(conscious_path))
    try:
        system = EmotionalMemorySystem(str(tmp_path / "state"))
        monkeypatch.setattr(
            system,
            "_latest_conscious_working_memory_seed",
            lambda: {
                "focus": "harbor_slope",
                "anchor": "harbor_slope",
                "strength": 0.4,
            },
        )
        summary = system._latest_nightly_working_memory_replay_summary()
    finally:
        if backup is None:
            nightly_path.unlink(missing_ok=True)
        else:
            nightly_path.write_text(backup, encoding="utf-8")
    assert summary is not None
    assert summary["focus"] == "market"
    assert summary["anchor"] == "market street"
    assert summary["conscious_memory_strength"] > 0.0


def test_latest_nightly_working_memory_replay_summary_reinforces_matching_conscious_seed(
    tmp_path, monkeypatch
) -> None:
    workspace = Path.cwd()
    reports_dir = workspace / "reports"
    reports_dir.mkdir(exist_ok=True)
    nightly_path = reports_dir / "nightly.json"
    backup = nightly_path.read_text(encoding="utf-8") if nightly_path.exists() else None
    nightly_path.write_text(
        json.dumps(
            {
                "inner_os_working_memory_replay_bias": {
                    "focus": "harbor_slope",
                    "anchor": "harbor_slope",
                    "strength": 0.4,
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    try:
        system = EmotionalMemorySystem(str(tmp_path / "state"))
        monkeypatch.setattr(
            system,
            "_latest_conscious_working_memory_seed",
            lambda: {
                "focus": "harbor_slope",
                "anchor": "harbor_slope",
                "strength": 0.5,
            },
        )
        summary = system._latest_nightly_working_memory_replay_summary()
    finally:
        if backup is None:
            nightly_path.unlink(missing_ok=True)
        else:
            nightly_path.write_text(backup, encoding="utf-8")
    assert summary is not None
    assert summary["focus"] == "harbor_slope"
    assert summary["anchor"] == "harbor_slope"
    assert summary["conscious_memory_overlap"] == 1.0
    assert summary["strength"] > 0.4
