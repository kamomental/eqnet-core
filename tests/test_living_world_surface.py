from fastapi.testclient import TestClient

from apps.observer.main import app


client = TestClient(app)


def test_living_world_surface_uses_runtime_script() -> None:
    response = client.get("/living-world")
    assert response.status_code == 200
    assert "/static/living_world_runtime.js" in response.text
    assert '/project-atri/2d-state' in response.text
    assert '/project-atri/2d-event' in response.text


def test_living_world_runtime_state_and_event_routes() -> None:
    state_response = client.get("/project-atri/2d-state")
    assert state_response.status_code == 200
    state_payload = state_response.json()
    assert state_payload["status"] == "ok"
    assert state_payload["state"]["schema"] == "project_atri_2d_state/v1"
    assert "sensing" in state_payload["state"]
    assert "voice_level" in state_payload["state"]["sensing"]

    event_response = client.post(
        "/project-atri/2d-event",
        json={
            "schema": "project_atri_2d_event/v1",
            "source": "test",
            "event_type": "stream_stage_enter",
            "world_id": "harbor_town",
            "payload": {"zone_id": "stream_stage", "world_type": "stage"},
        },
    )
    assert event_response.status_code == 200
    event_payload = event_response.json()
    assert event_payload["status"] == "ok"
    assert event_payload["state"]["identity"]["mode"] == "streaming"
