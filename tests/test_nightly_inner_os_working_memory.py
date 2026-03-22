import json
from pathlib import Path

from ops import nightly


def test_resolve_inner_os_working_memory_snapshot_reads_default_state_path(tmp_path, monkeypatch) -> None:
    state_dir = tmp_path / "data" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = state_dir / "inner_os_working_memory_snapshot.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "schema": "inner_os_working_memory_snapshot/v1",
                "snapshot": {"current_focus": "meaning", "promotion_readiness": 0.64},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    path, payload, warning = nightly._resolve_inner_os_working_memory_snapshot({})

    assert warning is None
    assert Path(path).resolve() == snapshot_path.resolve()
    assert isinstance(payload, dict)
    assert payload["snapshot"]["current_focus"] == "meaning"


def test_resolve_inner_os_working_memory_snapshot_warns_on_schema_mismatch(tmp_path, monkeypatch) -> None:
    state_dir = tmp_path / "data" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = state_dir / "inner_os_working_memory_snapshot.json"
    snapshot_path.write_text(
        json.dumps({"schema": "unexpected", "snapshot": {"current_focus": "body"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    path, payload, warning = nightly._resolve_inner_os_working_memory_snapshot({})

    assert path is None
    assert payload is None
    assert warning is not None
    assert "schema_mismatch" in warning
