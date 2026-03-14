from fastapi.testclient import TestClient

from apps.observer.main import app
from inner_os.http_contract import build_inner_os_manifest


client = TestClient(app)


def test_build_inner_os_manifest_contains_all_hooks() -> None:
    manifest = build_inner_os_manifest()
    assert manifest["schema"] == "inner_os_http_manifest/v1"
    assert set(manifest["hooks"].keys()) == {
        "pre_turn_update",
        "memory_recall",
        "response_gate",
        "post_turn_update",
    }
    assert manifest["hooks"]["memory_recall"]["recall_payload_schema"] == "inner_os_recall_payload/v1"
    assert manifest["contracts"]["recall_payload"]["required_fields"] == ["memory_anchor", "summary", "record_kind", "record_provenance"]


def test_inner_os_manifest_route() -> None:
    response = client.get("/inner-os/manifest")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["manifest"]["service"] == "inner_os"
    assert payload["manifest"]["hooks"]["pre_turn_update"]["path"] == "/inner-os/pre-turn-update"
    assert payload["manifest"]["contracts"]["memory_record"]["schema"] == "inner_os_memory/v1"
