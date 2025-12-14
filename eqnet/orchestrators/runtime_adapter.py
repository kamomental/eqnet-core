from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from eqnet.contracts import ActionIntent, PerceptPacket
from eqnet.orchestrators.common import (
    apply_trace_observations,
    build_percept_from_payload,
    extract_trace_observations,
    record_mapper_metadata,
)
from eqnet.runtime.turn import CoreState, SafetyConfig, run_turn
from eqnet.schemas.workspace import (
    FieldState,
    GoalWeight,
    MemoryReference,
    PerceptionBlock,
    RhythmDual,
    RhythmSingle,
    RhythmSnapshot,
    WorkspaceActionCandidate,
    WorkspaceActionSelection,
    WorkspaceSnapshot,
    WorkspaceState,
    WorkspaceValueState,
)
from eqnet.telemetry.trace_writer import append_trace_event, write_trace_jsonl

META_TEMPLATE = {
    "schema_version": "trace_v1",
    "source_loop": "runtime",
}


def runtime_payload_to_percept(payload: Mapping[str, object]) -> PerceptPacket:
    return build_percept_from_payload(payload)


def run_runtime_turn(
    payload: Mapping[str, object],
    state: CoreState,
    safety: SafetyConfig,
    trace_path: Path | str,
) -> None:
    """Run the runtime loop via the shared core and emit trace JSONL."""

    percept = runtime_payload_to_percept(payload)
    result = run_turn(percept, state, safety)
    apply_trace_observations(result.trace, extract_trace_observations(payload), source="runtime")
    record_mapper_metadata(result.trace)
    result.trace.source_loop = "runtime"
    result.trace.scenario_id = percept.scenario_id
    result.trace.turn_id = percept.turn_id
    result.trace.seed = percept.seed
    result.trace.timestamp_ms = percept.timestamp_ms
    meta = dict(META_TEMPLATE)
    meta["scenario_id"] = percept.scenario_id
    meta["turn_id"] = percept.turn_id
    meta["seed"] = percept.seed
    meta["timestamp_ms"] = percept.timestamp_ms
    trace_target = Path(trace_path)
    write_trace_jsonl(trace_target, result, meta=meta)
    snapshot = _build_workspace_snapshot(percept, result)
    append_trace_event(trace_target, snapshot)


def _clamp(value: float | None, *, lo: float = 0.0, hi: float = 1.0) -> float:
    if value is None:
        return lo
    try:
        val = float(value)
    except (TypeError, ValueError):
        return lo
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val


def _build_workspace_snapshot(percept: PerceptPacket, result) -> dict:
    timestamp_ms = percept.timestamp_ms or 0
    hazard = _clamp(percept.world.hazard_level)
    ambiguity = _clamp(percept.world.ambiguity)
    arousal = _clamp(percept.somatic.arousal_hint)
    stress = _clamp(percept.somatic.stress_hint)
    winner = str(result.trace.self_state.get("winner", "unknown")) if result.trace.self_state else "unknown"
    boundary_score = float((result.trace.boundary or {}).get("score", hazard))
    gate_level = 1 if boundary_score >= float((result.trace.boundary or {}).get("threshold", safety.boundary_threshold if isinstance(safety, SafetyConfig) else 0.7)) else 0

    field_state = FieldState(
        S_norm=stress,
        H_norm=hazard,
        rho_norm=1.0 - ambiguity,
        Ignition=float(boundary_score),
        valence=max(-1.0, min(1.0, float(percept.world.npc_affect))),
        arousal=arousal,
        gate_state=winner,
        gate_level=gate_level,
        S_raw=float(percept.world.ambiguity),
        H_raw=float(percept.world.hazard_level),
    )

    rhythm_self = RhythmSingle(R=_clamp(result.trace.boundary.get("score", hazard) if result.trace.boundary else hazard), psi=None, rho=hazard, I=float(boundary_score), q=None)
    rhythm_eff = RhythmSingle(R=rhythm_self.R, psi=None, rho=rhythm_self.rho, I=rhythm_self.I, q=None)
    rhythm_snapshot = RhythmSnapshot(self=rhythm_self, eff=rhythm_eff, other=None, dual=RhythmDual(dpsi=None, mismatch=None))

    memory_ref = MemoryReference(replayed=False)

    perception_block = PerceptionBlock(
        vision={"fog": ambiguity, "confidence": 1.0 - ambiguity},
        audio={"confidence": 1.0},
        interoception={
            "arousal_hint": percept.somatic.arousal_hint,
            "stress_hint": percept.somatic.stress_hint,
            "fatigue_hint": percept.somatic.fatigue_hint,
        },
    )

    goals = [GoalWeight(name="safety", weight=hazard), GoalWeight(name="explore", weight=1.0 - hazard)]
    value_state = WorkspaceValueState(goals=goals, risk=hazard)

    action_candidate = WorkspaceActionCandidate(
        name=result.intent.action if isinstance(result.intent, ActionIntent) else "unknown",
        score=_clamp(result.intent.confidence if isinstance(result.intent, ActionIntent) else 0.5),
        reason_codes=list((result.intent.rationale or {}).keys()) if isinstance(result.intent, ActionIntent) else [],
    )
    selected = WorkspaceActionSelection(
        name=action_candidate.name,
        expected_effect={k: v for k, v in (result.intent.rationale or {}).items()} if isinstance(result.intent, ActionIntent) else {},
    )
    workspace_state = WorkspaceState(
        perception=perception_block,
        value=value_state,
        action_candidates=[action_candidate],
        selected_action=selected,
    )

    snapshot_model = WorkspaceSnapshot(
        timestamp_ms=timestamp_ms,
        stage="runtime",
        step=timestamp_ms,
        source_loop="runtime",
        pid=os.getpid(),
        field=field_state,
        rhythm=rhythm_snapshot,
        memory_ref=memory_ref,
        workspace=workspace_state,
        control_out=None,
        intervention=None,
        meta={
            "turn_id": percept.turn_id,
            "scenario_id": percept.scenario_id,
            "tags": percept.tags,
        },
    )
    return snapshot_model.model_dump(mode="json")


__all__ = ["run_runtime_turn", "runtime_payload_to_percept"]
