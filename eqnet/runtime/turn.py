from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from eqnet.contracts import (
    ActionIntent,
    MemoryDelta,
    PerceptPacket,
    SpeechPlan,
    TraceSchema,
    TurnResult,
)
from eqnet.runtime.policy import PolicyPrior

try:
    from eqnet.runtime.state import QualiaState
except Exception:  # pragma: no cover
    QualiaState = Any  # type: ignore


@dataclass
class CoreState:
    policy_prior: PolicyPrior = field(default_factory=PolicyPrior)
    qualia: QualiaState | Dict[str, Any] | None = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyConfig:
    boundary_threshold: float = 0.7
    max_directiveness_in_hazard: float = 0.3
    prospection_jerk_reject_threshold: float = 0.8
    prospection_temp_reject_threshold: float = 0.9
    text_policy: str = "redact"
    text_truncate_chars: int = 200


def _qualia_to_dict(qualia: Any) -> Dict[str, Any]:
    if hasattr(qualia, "__dict__"):
        return dict(qualia.__dict__)
    if isinstance(qualia, dict):
        return dict(qualia)
    return {}


def run_turn(percept: PerceptPacket, state: CoreState, safety: SafetyConfig) -> TurnResult:
    """Pure-ish turn evaluation that records reasoning into the trace."""

    trace = TraceSchema(
        turn_id=percept.turn_id,
        seed=percept.seed,
        timestamp_ms=percept.timestamp_ms,
    )

    hazard = float(percept.world.hazard_level)
    boundary_score = max(0.0, min(1.0, hazard))
    trace.boundary = {
        "score": boundary_score,
        "threshold": safety.boundary_threshold,
        "reasons": {
            "hazard_level": hazard,
            "proximity": percept.somatic.proximity,
            "stress_hint": percept.somatic.stress_hint,
        },
    }

    fatigue = float(percept.somatic.fatigue_hint)
    is_hazard = boundary_score >= safety.boundary_threshold
    winner = "affective" if is_hazard else "narrative"
    trace.self_state = {
        "winner": winner,
        "tie_flag": False,
        "margin": 0.0,
        "fatigue": fatigue,
    }

    trace.prospection = {
        "jerk": 0.0,
        "temperature": safety.prospection_temp_reject_threshold,
        "accepted": False,
    }

    prior = state.policy_prior if isinstance(state.policy_prior, PolicyPrior) else PolicyPrior()
    throttles = {
        "offer_requested": percept.context.offer_requested,
        "cultural_pressure": percept.context.cultural_pressure,
        "directiveness_cap": safety.max_directiveness_in_hazard if is_hazard else 1.0,
    }
    trace.policy = {
        "prior": prior.__dict__ if hasattr(prior, "__dict__") else str(prior),
        "throttles": throttles,
    }

    qualia_snapshot = _qualia_to_dict(state.qualia)
    trace.qualia = {
        "before": qualia_snapshot,
        "after": qualia_snapshot,
    }

    intent = ActionIntent(
        action="speak",
        confidence=0.5,
        rationale={"winner": winner, "boundary": boundary_score},
    )
    speech = SpeechPlan(
        text="(stub) hello",
        tone="calm" if is_hazard else "neutral",
        directiveness=0.0,
        disclosure=0.0,
    )
    memory = MemoryDelta(
        should_write_diary=is_hazard,
        diary_tags=["boundary"] if is_hazard else [],
        salience=boundary_score,
    )

    return TurnResult(turn_id=percept.turn_id, intent=intent, speech=speech, memory=memory, trace=trace)


def update_state(state: CoreState, result: TurnResult) -> CoreState:  # noqa: ARG001
    """Apply ``result`` deltas into ``state`` (placeholder)."""

    return state


__all__ = ["run_turn", "update_state", "CoreState", "SafetyConfig"]


