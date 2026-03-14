from fastapi.testclient import TestClient

from inner_os.http_app import app


client = TestClient(app)


def test_inner_os_standalone_manifest_route() -> None:
    response = client.get("/inner-os/manifest")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["manifest"]["service"] == "inner_os"


def test_inner_os_standalone_pre_turn_route() -> None:
    response = client.post(
        "/inner-os/pre-turn-update",
        json={
            "user_input": {"text": "hello"},
            "sensor_input": {"voice_level": 0.35, "body_stress_index": 0.28},
            "local_context": {"last_gate_context": {"valence": 0.1, "arousal": 0.2}},
            "current_state": {"current_energy": 0.8},
            "safety_bias": 0.1,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["result"]["state"]["stress"] >= 0.0
