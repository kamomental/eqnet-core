from pathlib import Path

from fastapi.testclient import TestClient

from scripts import vision_bridge
from scripts.vision_frame_contract import build_sensor_frame


class DummyBridge:
    def __init__(self) -> None:
        self.last_sensor_frame = None

    def process_image(self, **kwargs):
        self.last_sensor_frame = kwargs.get("sensor_frame")
        return {
            "response_route": "watch",
            "response": {
                "text": "ok",
                "perception_summary": None,
                "retrieval_summary": None,
            },
        }

    def get_2d_state(self):
        return {"schema": "project_atri_2d_state/v1"}

    def ingest_2d_event(self, payload):
        return {"status": "accepted", "event_type": payload.get("event_type")}


client = TestClient(vision_bridge.app)


def test_build_sensor_frame_merges_explicit_sensor_fields() -> None:
    frame = build_sensor_frame(
        {
            "sensor": {
                "person_count": 2,
                "object_counts": {"cup": 1},
                "voice_level": 0.35,
                "motion_score": 0.2,
                "has_face": True,
                "privacy_tags": ["home"],
            }
        }
    )
    assert frame is not None
    assert frame["person_count"] == 2
    assert frame["object_counts"]["cup"] == 1
    assert frame["has_face"] is True
    assert frame["voice_level"] == 0.35


def test_build_sensor_frame_merges_audio_body_and_place_channels() -> None:
    frame = build_sensor_frame(
        {
            "sensor": {
                "audio": {"voice_level": 0.4, "breath_rate": 0.3},
                "body": {"heart_rate_raw": 86, "heart_rate_baseline": 72, "body_stress_index": 0.58},
                "place": {"place_id": "harbor_stage", "privacy_tags": ["public"]},
                "detections": [{"class": "person"}, {"class": "lamp"}],
            }
        }
    )
    assert frame is not None
    assert frame["voice_level"] == 0.4
    assert frame["breath_rate"] == 0.3
    assert frame["heart_rate_raw"] == 86.0
    assert frame["body_stress_index"] == 0.58
    assert frame["place_id"] == "harbor_stage"
    assert frame["privacy_tags"] == ["public"]
    assert frame["person_count"] == 1
    assert frame["object_counts"]["lamp"] == 1


def test_vision_frame_accepts_sensor_metadata(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"frame")
    bridge = DummyBridge()
    monkeypatch.setattr(vision_bridge, "get_runtime_bridge", lambda: bridge)

    response = client.post(
        "/vision-frame",
        json={
            "image_path": str(image_path),
            "prompt": "observe",
            "sensor": {
                "audio": {"voice_level": 0.45, "breath_rate": 0.2},
                "body": {"heart_rate_raw": 90, "heart_rate_baseline": 74, "motion_score": 0.1},
                "place": {"place_id": "harbor_stage", "privacy_tags": ["public"]},
                "person_count": 1,
                "object_counts": {"lamp": 1},
                "has_face": True,
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["sensor_schema"] == "vision_sensor_frame/v1"
    assert payload["sensor_summary"]["available"] is True
    assert payload["sensor_summary"]["person_count"] == 1
    assert payload["sensor_summary"]["place_id"] == "harbor_stage"
    assert payload["sensor_summary"]["privacy_tags"] == ["public"]
    assert bridge.last_sensor_frame is not None
    assert bridge.last_sensor_frame["object_counts"]["lamp"] == 1
    assert bridge.last_sensor_frame["heart_rate_raw"] == 90.0
