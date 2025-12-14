from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from eqnet.hub.api import EQNetConfig, EQNetHub


def _dummy_moment(ts: datetime | None = None) -> Any:
    timestamp = ts or datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    moment = SimpleNamespace()
    moment.timestamp = timestamp
    moment.awareness_stage = 1
    moment.emotion = SimpleNamespace(mask=0.1, love=0.3, stress=0.4, heart_rate_norm=0.5, breath_ratio_norm=0.6)
    moment.culture = SimpleNamespace(rho=0.2, politeness=0.7, intimacy=0.4)
    moment.mood = {"arousal": 0.5, "stress": 0.2}
    moment.metrics = {"proximity": 0.8, "stress": 0.2}
    moment.gate_context = {"mode": "talk", "cultural_pressure": 0.1}
    moment.session_id = "session-demo"
    moment.turn_id = 5
    moment.talk_mode = "talk"
    moment.emotion_tag = "joy"
    return moment


def _hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EQNetHub:
    config = EQNetConfig(
        telemetry_dir=tmp_path / "telemetry",
        reports_dir=tmp_path / "reports",
        state_dir=tmp_path / "state",
    )
    monkeypatch.setenv("EQNET_TRACE_V1", "1")
    return EQNetHub(config=config, embed_text_fn=lambda text: [0.0, 0.1, 0.2])


def test_log_moment_emits_trace_when_flag_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trace_root = tmp_path / "trace_dump"
    monkeypatch.setenv("EQNET_TRACE_V1_DIR", str(trace_root))
    hub = _hub(tmp_path, monkeypatch)
    hub.log_moment(_dummy_moment(), "hello user")
    day_dir = trace_root / "2024-01-02"
    files = list(day_dir.glob("hub-*.jsonl"))
    assert files, "trace output should be written when flag is enabled"
    data = json.loads(files[0].read_text(encoding="utf-8").strip())
    assert data["schema_version"] == "trace_v1"
    assert data["source_loop"] == "hub"
    assert data["scenario_id"] == "session-demo"
    assert set(data.keys()) >= {"boundary", "policy", "prospection", "qualia", "invariants"}
    qualia_obs = data.get("qualia", {}).get("observations", {}).get("hub", {})
    user_text_meta = qualia_obs.get("user_text") or {}
    assert user_text_meta.get("policy") == "redact"
    policy_obs = data.get("policy", {}).get("observations", {}).get("hub", {})
    assert policy_obs.get("talk_mode") == "talk"


def test_log_moment_does_not_fail_if_trace_emit_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = _hub(tmp_path, monkeypatch)
    monkeypatch.setenv("EQNET_TRACE_V1_DIR", str(tmp_path / "trace_dump"))

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("eqnet.hub.api.run_hub_turn", boom)
    hub.log_moment(_dummy_moment(), "still safe")
