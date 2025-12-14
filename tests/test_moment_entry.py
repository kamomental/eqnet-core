from dataclasses import dataclass

from eqnet.hub.moment_entry import to_moment_entry


@dataclass
class DemoMomentLogEntry:
    turn_id: str
    timestamp_ms: int
    user_text: str
    somatic: dict
    context: dict
    world: dict
    mood: dict
    metrics: dict
    gate_context: dict
    culture: dict
    emotion: dict


def test_to_moment_entry_dataclass_thin_extract():
    obj = DemoMomentLogEntry(
        turn_id="turn-1",
        timestamp_ms=123,
        user_text="secret@example.com 1234",
        somatic={"stress_hint": 0.0},
        context={"mode": "casual"},
        world={"hazard_level": 0.1},
        mood={"stress": 0.5},
        metrics={"proximity": 0.8},
        gate_context={"request": "advice"},
        culture={"rho": 0.2},
        emotion={"mask": 0.1},
    )
    entry = to_moment_entry(obj)
    assert entry["turn_id"] == "turn-1"
    assert entry["timestamp_ms"] == 123
    assert entry["somatic"]["stress_hint"] == 0.0
    assert entry["context"]["mode"] == "casual"
    assert entry["extras"]["raw_ref"]["type"] == "DemoMomentLogEntry"
