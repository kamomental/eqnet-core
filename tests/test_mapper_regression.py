from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from eqnet.orchestrators.common import build_percept_from_payload
from eqnet.orchestrators.hub_adapter import run_hub_turn
from eqnet.orchestrators.runtime_adapter import run_runtime_turn
from eqnet.policy.core_invariants import evaluate_core_invariants, load_core_invariants
from eqnet.runtime.turn import CoreState, SafetyConfig

MANDATORY_KEYS = {
    "schema_version",
    "source_loop",
    "scenario_id",
    "turn_id",
    "seed",
    "timestamp_ms",
    "boundary",
    "self",
    "prospection",
    "policy",
    "qualia",
    "invariants",
}


def _fixture_payload(name: str) -> dict:
    path = Path("fixtures/payloads") / name
    return json.loads(path.read_text(encoding="utf-8"))


def _standardize_event(payload: dict, idx: int) -> dict:
    event = dict(payload)
    event.setdefault("scenario_id", f"fixture_{idx}")
    event.setdefault("turn_id", f"fixture-turn-{idx:04d}")
    event.setdefault("timestamp_ms", (idx + 1) * 1000)
    event.setdefault("seed", idx + 42)
    return event


def _read_trace(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_mapper_regression_traces(tmp_path):
    events = [
        _standardize_event(_fixture_payload("hub_sample_001.json"), 0),
        _standardize_event(_fixture_payload("runtime_sample_001.json"), 1),
    ]
    hub_trace = tmp_path / "hub.jsonl"
    runtime_trace = tmp_path / "runtime.jsonl"
    safety = SafetyConfig()
    for event in events:
        run_hub_turn(event, CoreState(), safety, hub_trace)
        run_runtime_turn(event, CoreState(), safety, runtime_trace)

    hub_records = _read_trace(hub_trace)
    runtime_records = _read_trace(runtime_trace)
    invariants = load_core_invariants()

    for record in hub_records + runtime_records:
        for key in MANDATORY_KEYS:
            assert key in record
        evaluation = evaluate_core_invariants(record, invariants)
        assert evaluation.get("CORE_TRACE_001") is True
        boundary = record.get("boundary", {})
        reasons = boundary.get("reasons", {})
        hazard = reasons.get("hazard_level")
        if hazard is not None:
            assert 0.0 <= float(hazard) <= 1.0
        mapper_obs = record.get("policy", {}).get("observations", {}).get("mapper")
        assert mapper_obs is not None


def test_compare_loops_cli_directory(tmp_path):
    out_dir = tmp_path / "artifacts"
    cmd = [
        sys.executable,
        "-m",
        "eqnet.dev.compare_loops",
        "fixtures/payloads",
        "--out",
        str(out_dir),
        "--deterministic",
        "--strict-pairs",
    ]
    subprocess.run(cmd, check=True)
    diff_payload = json.loads((out_dir / "trace_diff.json").read_text(encoding="utf-8"))
    assert diff_payload["scenarios"], "scenarios should not be empty"
    assert diff_payload["aggregate"].get("missing_pairs") == []
    invariant_summary = json.loads((out_dir / "invariant_summary.json").read_text(encoding="utf-8"))
    metric_summary = json.loads((out_dir / "metric_summary.json").read_text(encoding="utf-8"))
    assert set(invariant_summary.keys()) == {"hub", "runtime"}
    assert set(metric_summary.keys()) == {"hub", "runtime"}
    for loop in ("hub", "runtime"):
        summary = invariant_summary[loop]
        assert set(summary.keys()) == {"fatal", "warn"}
        for severity in ("fatal", "warn"):
            assert isinstance(summary[severity], dict)
            assert not summary[severity]


def test_distance_to_proximity_is_monotonic():
    base = {
        "turn_id": "monotonic",
        "timestamp_ms": 0,
        "scenario_id": "monotonic",
        "somatic": {},
        "context": {},
        "world": {},
    }
    prev = None
    for distance in [0.0, 0.5, 1.0, 2.0, 5.0]:
        payload = dict(base)
        payload["somatic"] = {"distance": distance}
        percept = build_percept_from_payload(payload)
        prox = percept.somatic.proximity
        if prev is not None:
            assert prox <= prev + 1e-8
        prev = prox


def test_zero_values_preserved():
    payload = {
        "turn_id": "zero",
        "timestamp_ms": 1,
        "somatic": {"stress_hint": 0.0},
        "context": {},
        "world": {"hazard_level": 0.0, "danger": 0.9},
    }
    percept = build_percept_from_payload(payload)
    assert percept.world.hazard_level == 0.0
    assert percept.somatic.stress_hint == 0.0
