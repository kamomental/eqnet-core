from __future__ import annotations

import json
from pathlib import Path

from eqnet.orchestrators.hub_adapter import run_hub_turn
from eqnet.orchestrators.runtime_adapter import run_runtime_turn
from eqnet.runtime.turn import CoreState, SafetyConfig
from tests.e2e.proof_utils import load_proof_scenario, prepare_event

_COMPARABLE_KEYS = ["boundary", "prospection", "policy", "qualia", "self", "invariants"]


def _write_trace(events, runner, trace_path: Path) -> list[dict]:
    state = CoreState()
    safety = SafetyConfig()
    for event in events:
        runner(event, state, safety, trace_path)
    return _read_trace(trace_path)


def _read_trace(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _strip_observations(section: dict | None) -> dict:
    data = dict(section or {})
    if isinstance(data.get("observations"), dict):
        data = dict(data)
        data.pop("observations", None)
    return data


def test_hub_and_runtime_traces_match(tmp_path: Path) -> None:
    scenario = load_proof_scenario()
    events = [
        prepare_event(raw, idx, scenario.scenario_id, scenario.date)
        for idx, raw in enumerate(scenario.events)
    ]

    hub_trace = tmp_path / "hub.jsonl"
    runtime_trace = tmp_path / "runtime.jsonl"

    hub_records = _write_trace(events, run_hub_turn, hub_trace)
    runtime_records = _write_trace(events, run_runtime_turn, runtime_trace)

    assert len(hub_records) == len(runtime_records) > 0

    for hub_record, runtime_record in zip(hub_records, runtime_records):
        assert hub_record["turn_id"] == runtime_record["turn_id"]
        assert hub_record["source_loop"] == "hub"
        assert runtime_record["source_loop"] == "runtime"
        assert hub_record.get("user_text") == runtime_record.get("user_text")
        for key in _COMPARABLE_KEYS:
            hub_section = _strip_observations(hub_record.get(key))
            runtime_section = _strip_observations(runtime_record.get(key))
            assert hub_section == runtime_section
