"""World transition helpers for HeartOS."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional


@dataclass(frozen=True)
class TransitionParams:
    decay: float = 0.6
    uncertainty_factor: float = 0.5
    base_uncertainty: float = 0.2
    reason: str = "world_transition"


def apply_transition(
    state: Mapping[str, Any], params: TransitionParams
) -> MutableMapping[str, Any]:
    """Apply reset/retain rules to a minimal state dict."""
    drive = float(state.get("drive", 0.0))
    uncertainty = float(state.get("uncertainty", params.base_uncertainty))
    drive = max(0.0, drive * params.decay)
    uncertainty = max(params.base_uncertainty, uncertainty * params.uncertainty_factor)
    return {
        "drive": drive,
        "uncertainty": uncertainty,
        "hazard_sources": [],
    }


def build_transition_record(
    *,
    turn_id: str,
    transition_turn_index: int,
    scenario_id: str,
    from_world: str,
    to_world: str,
    params: TransitionParams,
    timestamp_ms: Optional[int] = None,
    source_loop: str = "world_transition",
) -> Dict[str, Any]:
    """Build a trace_v1 record for a world transition event."""
    ts_ms = int(time.time() * 1000) if timestamp_ms is None else int(timestamp_ms)
    return {
        "schema_version": "trace_v1",
        "timestamp_ms": ts_ms,
        "turn_id": turn_id,
        "scenario_id": scenario_id,
        "source_loop": source_loop,
        "event_type": "world_transition",
        "boundary": {"score": 0.0, "reasons": {}},
        "prospection": {"accepted": True, "jerk": 0.0, "temperature": 0.0},
        "policy": {"throttles": {}},
        "invariants": {},
        "transition": {
            "from_world_type": from_world,
            "to_world_type": to_world,
            "reason": params.reason,
            "decay": params.decay,
            "uncertainty_factor": params.uncertainty_factor,
            "base_uncertainty": params.base_uncertainty,
            "transition_turn_index": int(transition_turn_index),
        },
    }


__all__ = ["TransitionParams", "apply_transition", "build_transition_record"]
