from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Spin up the FastAPI app with tmp telemetry data."""

    audit_dir = tmp_path / "telemetry" / "audit"
    trace_dir = tmp_path / "telemetry" / "trace_v1" / "2025-12-13"
    audit_dir.mkdir(parents=True)
    trace_dir.mkdir(parents=True)

    audit_payload = {
        "metadata": {"date": "2025-12-13", "schema_version": "nightly_audit_v1"},
        "health": {"status": "YELLOW", "reasons": [{"code": "W001", "message": "warn"}]},
        "stats": {"fatal_failures": 0, "warn_fail_rate": 0.12},
        "boundary": {"summary": {"max_length": 12}},
    }
    (audit_dir / "nightly_audit_2025-12-13.json").write_text(json.dumps(audit_payload), encoding="utf-8")

    trace_lines = [
        json.dumps({
            "timestamp_ms": 1_000,
            "turn_id": "1",
            "pid": 1234,
            "source_loop": "runtime",
            "raw_transcript": "hello",
        }),
        json.dumps({"timestamp_ms": 2_000, "turn_id": "2", "raw_transcript": "world"}),
        json.dumps({"timestamp_ms": 3_000, "turn_id": "250", "raw_transcript": "!"}),
        "{\"timestamp_ms\": 4_000, \"turn_id\": \"broken",
    ]
    (trace_dir / "runtime-abc-1234.jsonl").write_text("\n".join(trace_lines), encoding="utf-8")

    from apps.observer.stores.audit_store import AuditStore
    from apps.observer.stores.trace_store import TraceStore
    from apps.observer.services.observer_service import ObserverService

    service = ObserverService(
        audit_store=AuditStore(audit_dir),
        trace_store=TraceStore(trace_dir.parent),
    )

    import apps.observer.deps as deps

    monkeypatch.setattr(deps, "_svc", service, raising=True)
    monkeypatch.setattr(deps, "svc", lambda: service, raising=True)

    from apps.observer.main import app

    return TestClient(app)
