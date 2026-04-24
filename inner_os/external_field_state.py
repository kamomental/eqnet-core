from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ExternalFieldFrame:
    """外部場の最近フレーム。"""

    step: int = 0
    dominant_field: str = "open_field"
    social_mode: str = "ambient"
    thread_mode: str = ""
    environmental_load: float = 0.0
    social_pressure: float = 0.0
    continuity_pull: float = 0.0
    ambiguity_load: float = 0.0
    safety_envelope: float = 0.0
    novelty: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "dominant_field": self.dominant_field,
            "social_mode": self.social_mode,
            "thread_mode": self.thread_mode,
            "environmental_load": round(self.environmental_load, 4),
            "social_pressure": round(self.social_pressure, 4),
            "continuity_pull": round(self.continuity_pull, 4),
            "ambiguity_load": round(self.ambiguity_load, 4),
            "safety_envelope": round(self.safety_envelope, 4),
            "novelty": round(self.novelty, 4),
        }


@dataclass(frozen=True)
class ExternalFieldState:
    """環境・関係・継続の外部場を束ねる canonical state。"""

    dominant_field: str = "open_field"
    social_mode: str = "ambient"
    thread_mode: str = ""
    environmental_load: float = 0.0
    social_pressure: float = 0.0
    continuity_pull: float = 0.0
    ambiguity_load: float = 0.0
    safety_envelope: float = 0.0
    novelty: float = 0.0
    trace: tuple[ExternalFieldFrame, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dominant_field": self.dominant_field,
            "social_mode": self.social_mode,
            "thread_mode": self.thread_mode,
            "environmental_load": round(self.environmental_load, 4),
            "social_pressure": round(self.social_pressure, 4),
            "continuity_pull": round(self.continuity_pull, 4),
            "ambiguity_load": round(self.ambiguity_load, 4),
            "safety_envelope": round(self.safety_envelope, 4),
            "novelty": round(self.novelty, 4),
            "trace": [frame.to_dict() for frame in self.trace],
        }

    def to_packet_axes(
        self,
        previous: Mapping[str, Any] | "ExternalFieldState" | None = None,
    ) -> dict[str, dict[str, float]]:
        previous_state = coerce_external_field_state(previous)
        current_axes = _axis_values(self)
        previous_axes = _axis_values(previous_state)
        return {
            axis_name: {
                "value": round(axis_value, 4),
                "delta": round(axis_value - previous_axes.get(axis_name, 0.0), 4),
            }
            for axis_name, axis_value in current_axes.items()
        }


def derive_external_field_state(
    *,
    previous_state: Mapping[str, Any] | ExternalFieldState | None = None,
    environment_pressure: Mapping[str, Any] | None = None,
    social_topology_state: Mapping[str, Any] | None = None,
    relation_competition_state: Mapping[str, Any] | None = None,
    recent_dialogue_state: Mapping[str, Any] | None = None,
    discussion_thread_state: Mapping[str, Any] | None = None,
    issue_state: Mapping[str, Any] | None = None,
    transition_signal: Mapping[str, Any] | None = None,
    organism_state: Mapping[str, Any] | None = None,
    trace_limit: int = 8,
) -> ExternalFieldState:
    previous = coerce_external_field_state(previous_state)
    environment = dict(environment_pressure or {})
    social_topology = dict(social_topology_state or {})
    relation = dict(relation_competition_state or {})
    recent = dict(recent_dialogue_state or {})
    discussion = dict(discussion_thread_state or {})
    issue = dict(issue_state or {})
    transition = dict(transition_signal or {})
    organism = dict(organism_state or {})

    resource_pressure = _float01(environment.get("resource_pressure"))
    hazard_pressure = _float01(environment.get("hazard_pressure"))
    ritual_pressure = _float01(environment.get("ritual_pressure"))
    institutional_pressure = _float01(environment.get("institutional_pressure"))
    social_density = _float01(environment.get("social_density"))
    visibility_pressure = _float01(social_topology.get("visibility_pressure"))
    threading_pressure = _float01(social_topology.get("threading_pressure"))
    hierarchy_pressure = _float01(social_topology.get("hierarchy_pressure"))
    competition_level = _float01(relation.get("competition_level"))
    dominant_score = _float01(relation.get("dominant_score"))
    overlap_score = _float01(recent.get("overlap_score"))
    reopen_pressure = _float01(recent.get("reopen_pressure"))
    thread_carry = _float01(recent.get("thread_carry"))
    unresolved_pressure = _float01(discussion.get("unresolved_pressure"))
    revisit_readiness = _float01(discussion.get("revisit_readiness"))
    thread_visibility = _float01(discussion.get("thread_visibility"))
    issue_question_pressure = _float01(issue.get("question_pressure"))
    issue_pause_readiness = _float01(issue.get("pause_readiness"))
    transition_intensity = _float01(transition.get("transition_intensity"))
    organism_grounding = _float01(organism.get("grounding"))
    organism_relation_pull = _float01(organism.get("relation_pull"))

    environmental_load = _clamp01(
        resource_pressure * 0.18
        + hazard_pressure * 0.34
        + ritual_pressure * 0.16
        + institutional_pressure * 0.14
        + social_density * 0.18
    )
    social_pressure = _clamp01(
        visibility_pressure * 0.24
        + threading_pressure * 0.16
        + hierarchy_pressure * 0.18
        + social_density * 0.18
        + competition_level * 0.12
        + (0.12 if _text(social_topology.get("state")) == "public_visible" else 0.0)
        + (0.08 if _text(social_topology.get("state")) == "hierarchical" else 0.0)
    )
    continuity_pull = _clamp01(
        thread_carry * 0.24
        + reopen_pressure * 0.18
        + thread_visibility * 0.16
        + revisit_readiness * 0.16
        + issue_pause_readiness * 0.08
        + dominant_score * 0.08
        + organism_relation_pull * 0.1
    )
    ambiguity_load = _clamp01(
        issue_question_pressure * 0.26
        + unresolved_pressure * 0.22
        + overlap_score * 0.14
        + transition_intensity * 0.12
        + max(0.0, 1.0 - thread_visibility) * 0.1
        + competition_level * 0.08
        + hierarchy_pressure * 0.08
    )
    novelty = _clamp01(
        transition_intensity * 0.42
        + _bool_weight(transition.get("place_changed"), 0.18)
        + _bool_weight(transition.get("body_state_changed"), 0.1)
        + _bool_weight(transition.get("privacy_shift"), 0.16)
        + _bool_weight(transition.get("density_shift"), 0.14)
    )
    safety_envelope = _clamp01(
        0.52
        + (1.0 - hazard_pressure) * 0.16
        + (1.0 - social_pressure) * 0.08
        + thread_visibility * 0.06
        + organism_grounding * 0.1
        + organism_relation_pull * 0.06
        - novelty * 0.12
        - hierarchy_pressure * 0.08
    )

    environmental_load = _carry(previous.environmental_load, environmental_load, previous_state, 0.22)
    social_pressure = _carry(previous.social_pressure, social_pressure, previous_state, 0.2)
    continuity_pull = _carry(previous.continuity_pull, continuity_pull, previous_state, 0.24)
    ambiguity_load = _carry(previous.ambiguity_load, ambiguity_load, previous_state, 0.18)
    novelty = _carry(previous.novelty, novelty, previous_state, 0.12)
    safety_envelope = _carry(previous.safety_envelope, safety_envelope, previous_state, 0.18)

    dominant_field = _dominant_field(
        environmental_load=environmental_load,
        social_pressure=social_pressure,
        continuity_pull=continuity_pull,
        ambiguity_load=ambiguity_load,
        novelty=novelty,
        hazard_pressure=hazard_pressure,
        ritual_pressure=ritual_pressure,
        institutional_pressure=institutional_pressure,
    )
    social_mode = _text(social_topology.get("state")) or previous.social_mode or "ambient"
    thread_mode = (
        _text(discussion.get("state"))
        or _text(recent.get("state"))
        or _text(issue.get("state"))
        or previous.thread_mode
    )
    step = previous.trace[-1].step + 1 if previous.trace else 1
    trace = list(previous.trace[-max(0, trace_limit - 1) :]) if trace_limit > 0 else []
    trace.append(
        ExternalFieldFrame(
            step=step,
            dominant_field=dominant_field,
            social_mode=social_mode,
            thread_mode=thread_mode,
            environmental_load=environmental_load,
            social_pressure=social_pressure,
            continuity_pull=continuity_pull,
            ambiguity_load=ambiguity_load,
            safety_envelope=safety_envelope,
            novelty=novelty,
        )
    )
    return ExternalFieldState(
        dominant_field=dominant_field,
        social_mode=social_mode,
        thread_mode=thread_mode,
        environmental_load=environmental_load,
        social_pressure=social_pressure,
        continuity_pull=continuity_pull,
        ambiguity_load=ambiguity_load,
        safety_envelope=safety_envelope,
        novelty=novelty,
        trace=tuple(trace[-trace_limit:] if trace_limit > 0 else ()),
    )


def coerce_external_field_state(
    value: Mapping[str, Any] | ExternalFieldState | None,
) -> ExternalFieldState:
    if isinstance(value, ExternalFieldState):
        return value
    payload = dict(value or {})
    trace_items: list[ExternalFieldFrame] = []
    for item in payload.get("trace") or ():
        if isinstance(item, ExternalFieldFrame):
            trace_items.append(item)
        elif isinstance(item, Mapping):
            trace_items.append(
                ExternalFieldFrame(
                    step=int(_float(item.get("step"), 0.0)),
                    dominant_field=_text(item.get("dominant_field")) or "open_field",
                    social_mode=_text(item.get("social_mode")) or "ambient",
                    thread_mode=_text(item.get("thread_mode")),
                    environmental_load=_float01(item.get("environmental_load")),
                    social_pressure=_float01(item.get("social_pressure")),
                    continuity_pull=_float01(item.get("continuity_pull")),
                    ambiguity_load=_float01(item.get("ambiguity_load")),
                    safety_envelope=_float01(item.get("safety_envelope")),
                    novelty=_float01(item.get("novelty")),
                )
            )
    return ExternalFieldState(
        dominant_field=_text(payload.get("dominant_field")) or "open_field",
        social_mode=_text(payload.get("social_mode")) or "ambient",
        thread_mode=_text(payload.get("thread_mode")),
        environmental_load=_float01(payload.get("environmental_load")),
        social_pressure=_float01(payload.get("social_pressure")),
        continuity_pull=_float01(payload.get("continuity_pull")),
        ambiguity_load=_float01(payload.get("ambiguity_load")),
        safety_envelope=_float01(payload.get("safety_envelope")),
        novelty=_float01(payload.get("novelty")),
        trace=tuple(trace_items),
    )


def _axis_values(state: ExternalFieldState) -> dict[str, float]:
    return {
        "environment": _clamp01(state.environmental_load),
        "social": _clamp01(state.social_pressure),
        "continuity": _clamp01(state.continuity_pull),
        "ambiguity": _clamp01(state.ambiguity_load),
        "safety": _clamp01(state.safety_envelope),
        "novelty": _clamp01(state.novelty),
    }


def _dominant_field(
    *,
    environmental_load: float,
    social_pressure: float,
    continuity_pull: float,
    ambiguity_load: float,
    novelty: float,
    hazard_pressure: float,
    ritual_pressure: float,
    institutional_pressure: float,
) -> str:
    if ambiguity_load >= max(environmental_load, social_pressure, continuity_pull, novelty) and ambiguity_load >= 0.44:
        return "ambiguous_field"
    if continuity_pull >= max(environmental_load, social_pressure, ambiguity_load, novelty) and continuity_pull >= 0.42:
        return "continuity_field"
    if social_pressure >= max(environmental_load, ambiguity_load, continuity_pull, novelty) and social_pressure >= 0.42:
        return "social_pressure_field"
    if hazard_pressure >= max(ritual_pressure, institutional_pressure) and environmental_load >= 0.4:
        return "hazard_field"
    if max(ritual_pressure, institutional_pressure) >= 0.4:
        return "formal_field"
    if novelty >= 0.44:
        return "shifting_field"
    return "open_field"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _float01(value: Any, default: float = 0.0) -> float:
    return _clamp01(_float(value, default))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _carry(previous_value: float, current_value: float, previous_state: Any, alpha: float) -> float:
    if previous_state is None:
        return _clamp01(current_value)
    return _clamp01(previous_value * alpha + current_value * (1.0 - alpha))


def _bool_weight(value: Any, weight: float) -> float:
    return float(weight if bool(value) else 0.0)
