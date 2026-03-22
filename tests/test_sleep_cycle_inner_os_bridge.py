import json
from pathlib import Path

from emot_terrain_lab.sleep.cycle import nightly_consolidate


class _StubMem:
    def __init__(self) -> None:
        self.last_decay = None

    def decay(self, factor: float) -> None:
        self.last_decay = factor


class _StubSelfModel:
    def coherence(self) -> float:
        return 0.42


class _StubMemoryLayer:
    def __init__(self, size: int, attr: str) -> None:
        setattr(self, attr, [object()] * size)


class _StubLegacySystem:
    def __init__(self) -> None:
        self.l1 = _StubMemoryLayer(12, "experiences")
        self.l2 = _StubMemoryLayer(5, "episodes")
        self.l3 = _StubMemoryLayer(2, "patterns")

    def rest_state(self) -> dict:
        return {
            "active": True,
            "fatigue_streak": 2,
            "history": [
                {"triggers": {"fatigue": True, "loop": True, "overload": False}},
                {"triggers": {"fatigue": False, "loop": True, "overload": True}},
            ],
        }

    def field_metrics_state(self) -> list[dict]:
        return [{"entropy": 8.1, "enthalpy": 0.58}]


def test_nightly_consolidate_writes_inner_os_sleep_snapshot_when_legacy_system_present(tmp_path: Path) -> None:
    replay_path = tmp_path / "replay_firings.jsonl"
    replay_path.write_text(
        "\n".join(
            [
                json.dumps({"ts": 1000.0, "U_top": 0.3}),
                json.dumps({"ts": 2000.0, "U_top": 0.9}),
            ]
        ),
        encoding="utf-8",
    )
    summary_path = tmp_path / "nightly_summary.json"

    mem = _StubMem()
    summary = nightly_consolidate(
        replay_log_path=str(replay_path),
        mem=mem,
        self_model=_StubSelfModel(),
        legacy_system=_StubLegacySystem(),
        out_path=str(summary_path),
    )

    assert summary_path.exists()
    assert mem.last_decay is not None
    assert summary["inner_os_sleep_snapshot_path"]
    sleep_path = Path(str(summary["inner_os_sleep_snapshot_path"]))
    assert sleep_path.exists()
    payload = json.loads(sleep_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "inner_os_sleep_consolidation_snapshot/v1"
    assert summary["inner_os_sleep_mode"] == payload["snapshot"]["mode"]
