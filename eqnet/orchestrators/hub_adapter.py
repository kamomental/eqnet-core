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
from eqnet.telemetry.trace_writer import append_trace_event, write_trace_jsonl

META_TEMPLATE = {
    "schema_version": "trace_v1",
    "source_loop": "hub",
    "runtime_version": "unknown",
    "idempotency_key": "",
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
    _apply_forced_gate_action(result.trace)
    record_mapper_metadata(result.trace)
    result.trace.source_loop = "hub"
    result.trace.scenario_id = percept.scenario_id
    result.trace.turn_id = percept.turn_id
    result.trace.seed = percept.seed
    result.trace.timestamp_ms = percept.timestamp_ms
    meta = dict(META_TEMPLATE)
    runtime_version = event.get("runtime_version")
    if isinstance(runtime_version, str) and runtime_version.strip():
        meta["runtime_version"] = runtime_version.strip()
    idem_key = event.get("idempotency_key")
    if isinstance(idem_key, str):
        meta["idempotency_key"] = idem_key
    meta["scenario_id"] = percept.scenario_id
    meta["turn_id"] = percept.turn_id
    meta["seed"] = percept.seed
    meta["timestamp_ms"] = percept.timestamp_ms
    write_trace_jsonl(Path(trace_path), result, meta=meta)
    _emit_forced_gate_trace_event(
        trace_path=Path(trace_path),
        event=event,
        trace=result.trace,
    )


def _apply_forced_gate_action(trace: Any) -> None:
    policy = trace.policy if isinstance(getattr(trace, "policy", None), dict) else {}
    observations = policy.get("observations") if isinstance(policy, dict) else None
    hub_obs = observations.get("hub") if isinstance(observations, Mapping) else None
    forced = str(hub_obs.get("forced_gate_action") or "") if isinstance(hub_obs, Mapping) else ""
    if forced != "HUMAN_CONFIRM":
        return
    policy["gate_action"] = "HUMAN_CONFIRM"
    prospection = trace.prospection if isinstance(getattr(trace, "prospection", None), dict) else {}
    prospection["accepted"] = False
    if isinstance(hub_obs, dict):
        reason_codes = hub_obs.get("reason_codes")
        if not isinstance(reason_codes, list):
            reason_codes = []
        if "ONLINE_DELTA_FORCE_HUMAN_CONFIRM" not in reason_codes:
            reason_codes.append("ONLINE_DELTA_FORCE_HUMAN_CONFIRM")
        hub_obs["reason_codes"] = reason_codes


def _emit_forced_gate_trace_event(*, trace_path: Path, event: Mapping[str, Any], trace: Any) -> None:
    policy = trace.policy if isinstance(getattr(trace, "policy", None), dict) else {}
    observations = policy.get("observations") if isinstance(policy, dict) else None
    hub_obs = observations.get("hub") if isinstance(observations, Mapping) else None
    if not isinstance(hub_obs, Mapping):
        return
    forced = str(hub_obs.get("forced_gate_action") or "")
    if forced != "HUMAN_CONFIRM":
        return
    timestamp_ms = int(event.get("timestamp_ms") or trace.timestamp_ms or 0)
    if timestamp_ms <= 0:
        return
    reason_codes = [str(code) for code in (hub_obs.get("reason_codes") or []) if isinstance(code, str) and code]
    delta_ids = [str(item) for item in (hub_obs.get("online_delta_ids") or []) if isinstance(item, str) and item]
    forced_event = {
        "schema_version": "trace_v1",
        "source_loop": "hub",
        "scenario_id": event.get("scenario_id"),
        "turn_id": f"{event.get('turn_id')}-forced-gate",
        "seed": event.get("seed"),
        "timestamp_ms": timestamp_ms,
        "event_type": "forced_gate_action",
        "forced_gate_action": "HUMAN_CONFIRM",
        "reason_codes": reason_codes or ["ONLINE_DELTA_FORCE_HUMAN_CONFIRM"],
        "online_delta_ids": delta_ids,
        "boundary": {},
        "self": {},
        "prospection": {"accepted": False},
        "policy": {"observations": {"hub": {"forced_gate_action": "HUMAN_CONFIRM"}}},
        "qualia": {},
        "invariants": {},
    }
    append_trace_event(trace_path, forced_event)


__all__ = ["run_hub_turn", "event_to_percept"]
