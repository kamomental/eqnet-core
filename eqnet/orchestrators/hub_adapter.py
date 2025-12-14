from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from eqnet.contracts import PerceptPacket
from eqnet.orchestrators.common import (
    apply_trace_observations,
    build_percept_from_payload,
    extract_trace_observations,
    record_mapper_metadata,
)
from eqnet.runtime.turn import CoreState, SafetyConfig, run_turn
from eqnet.telemetry.trace_writer import write_trace_jsonl

META_TEMPLATE = {
    "schema_version": "trace_v1",
    "source_loop": "hub",
}


def event_to_percept(event: Mapping[str, Any]) -> PerceptPacket:
    return build_percept_from_payload(event)


def run_hub_turn(
    event: Mapping[str, Any],
    state: CoreState,
    safety: SafetyConfig,
    trace_path: Path | str,
) -> None:
    """Convert ``event`` into a percept, run the core loop, and emit trace JSONL."""

    percept = event_to_percept(event)
    result = run_turn(percept, state, safety)
    apply_trace_observations(result.trace, extract_trace_observations(event), source="hub")
    record_mapper_metadata(result.trace)
    result.trace.source_loop = "hub"
    result.trace.scenario_id = percept.scenario_id
    result.trace.turn_id = percept.turn_id
    result.trace.seed = percept.seed
    result.trace.timestamp_ms = percept.timestamp_ms
    meta = dict(META_TEMPLATE)
    meta["scenario_id"] = percept.scenario_id
    meta["turn_id"] = percept.turn_id
    meta["seed"] = percept.seed
    meta["timestamp_ms"] = percept.timestamp_ms
    write_trace_jsonl(Path(trace_path), result, meta=meta)


__all__ = ["run_hub_turn", "event_to_percept"]
