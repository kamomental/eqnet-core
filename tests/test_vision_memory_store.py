import json

from emot_terrain_lab.memory.vision_memory_store import VisionMemoryStore


def test_append_observed_writes_first_entry(tmp_path) -> None:
    store = VisionMemoryStore(path=tmp_path / "vision_memory.jsonl")
    entry = store.append_observed(
        perception_summary={"text": "Calm blue graph with light red spikes.", "response_id": "resp_a"},
        turn_id=1,
        session_id="session-1",
        talk_mode="watch",
        response_route="habit",
        image_path="frame.png",
        timestamp=10.0,
    )
    assert entry is not None
    rows = [json.loads(line) for line in (tmp_path / "vision_memory.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["schema"] == "observed_vision/v1"


def test_append_observed_suppresses_near_duplicate(tmp_path) -> None:
    store = VisionMemoryStore(path=tmp_path / "vision_memory.jsonl", dedupe_window_seconds=300.0)
    first = store.append_observed(
        perception_summary={"text": "Calm blue graph with light red spikes and clean grid.", "response_id": "resp_first"},
        turn_id=1,
        session_id="session-1",
        talk_mode="watch",
        response_route="habit",
        image_path="frame.png",
        timestamp=10.0,
    )
    second = store.append_observed(
        perception_summary={"text": "Calm blue graph with red spikes and a clean grid on white background.", "response_id": "resp_second"},
        turn_id=2,
        session_id="session-1",
        talk_mode="watch",
        response_route="habit",
        image_path="frame.png",
        timestamp=20.0,
    )
    rows = [json.loads(line) for line in (tmp_path / "vision_memory.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert first is not None
    assert second is not None
    assert second["id"] == first["id"]
    assert second["suppressed"] is True
    assert second["meta"]["duplicate_suppressed"] is True


def test_append_observed_keeps_distinct_entry(tmp_path) -> None:
    store = VisionMemoryStore(path=tmp_path / "vision_memory.jsonl", dedupe_window_seconds=300.0)
    store.append_observed(
        perception_summary={"text": "Calm blue graph with light red spikes and clean grid.", "response_id": "resp_first"},
        turn_id=1,
        session_id="session-1",
        talk_mode="watch",
        response_route="habit",
        image_path="frame.png",
        timestamp=10.0,
    )
    distinct = store.append_observed(
        perception_summary={"text": "Harbor street scene with two people, stalls, and a bridge under warm light.", "response_id": "resp_second"},
        turn_id=2,
        session_id="session-1",
        talk_mode="talk",
        response_route="habit",
        image_path="street.png",
        timestamp=20.0,
    )
    rows = [json.loads(line) for line in (tmp_path / "vision_memory.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    assert distinct is not None
    assert distinct.get("suppressed") is None
