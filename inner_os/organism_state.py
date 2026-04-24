from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .epistemic_state import coerce_epistemic_state
from .growth_state import coerce_growth_state
from .heartbeat_structure_state import coerce_heartbeat_structure_state
from .qualia_structure_state import coerce_qualia_structure_state


@dataclass(frozen=True)
class OrganismFrame:
    """時系列で観測する organism state の最小フレーム。"""

    step: int = 0
    dominant_posture: str = "steady"
    relation_focus: str = ""
    social_mode: str = "ambient"
    attunement: float = 0.0
    coherence: float = 0.0
    grounding: float = 0.0
    protective_tension: float = 0.0
    expressive_readiness: float = 0.0
    play_window: float = 0.0
    relation_pull: float = 0.0
    social_exposure: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "dominant_posture": self.dominant_posture,
            "relation_focus": self.relation_focus,
            "social_mode": self.social_mode,
            "attunement": round(self.attunement, 4),
            "coherence": round(self.coherence, 4),
            "grounding": round(self.grounding, 4),
            "protective_tension": round(self.protective_tension, 4),
            "expressive_readiness": round(self.expressive_readiness, 4),
            "play_window": round(self.play_window, 4),
            "relation_pull": round(self.relation_pull, 4),
            "social_exposure": round(self.social_exposure, 4),
        }


@dataclass(frozen=True)
class OrganismState:
    """既存 projection を束ねる小さな canonical latent。"""

    attunement: float = 0.0
    coherence: float = 0.0
    grounding: float = 0.0
    protective_tension: float = 0.0
    expressive_readiness: float = 0.0
    play_window: float = 0.0
    relation_pull: float = 0.0
    social_exposure: float = 0.0
    dominant_posture: str = "steady"
    relation_focus: str = ""
    social_mode: str = "ambient"
    trace: tuple[OrganismFrame, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "attunement": round(self.attunement, 4),
            "coherence": round(self.coherence, 4),
            "grounding": round(self.grounding, 4),
            "protective_tension": round(self.protective_tension, 4),
            "expressive_readiness": round(self.expressive_readiness, 4),
            "play_window": round(self.play_window, 4),
            "relation_pull": round(self.relation_pull, 4),
            "social_exposure": round(self.social_exposure, 4),
            "dominant_posture": self.dominant_posture,
            "relation_focus": self.relation_focus,
            "social_mode": self.social_mode,
            "trace": [frame.to_dict() for frame in self.trace],
        }

    def to_packet_axes(
        self,
        previous: Mapping[str, Any] | "OrganismState" | None = None,
    ) -> dict[str, dict[str, float]]:
        previous_state = coerce_organism_state(previous)
        current_axes = _axis_values(self)
        previous_axes = _axis_values(previous_state)
        return {
            axis_name: {
                "value": round(axis_value, 4),
                "delta": round(axis_value - previous_axes.get(axis_name, 0.0), 4),
            }
            for axis_name, axis_value in current_axes.items()
        }


def derive_organism_state(
    *,
    previous_state: Mapping[str, Any] | OrganismState | None,
    growth_state: Mapping[str, Any] | None,
    epistemic_state: Mapping[str, Any] | None,
    qualia_structure_state: Mapping[str, Any] | None,
    heartbeat_structure_state: Mapping[str, Any] | None,
    relation_competition_state: Mapping[str, Any] | None = None,
    social_topology_state: Mapping[str, Any] | None = None,
    trace_limit: int = 8,
) -> OrganismState:
    previous = coerce_organism_state(previous_state)
    growth = coerce_growth_state(growth_state)
    epistemic = coerce_epistemic_state(epistemic_state)
    qualia = coerce_qualia_structure_state(qualia_structure_state)
    heartbeat = coerce_heartbeat_structure_state(heartbeat_structure_state)
    relation = dict(relation_competition_state or {})
    social = dict(social_topology_state or {})

    relation_focus = _text(relation.get("dominant_person_id"))
    social_mode = _text(social.get("state")) or "ambient"
    dominant_score = _float01(relation.get("dominant_score"))
    competition_level = _float01(relation.get("competition_level"))
    visibility_pressure = _float01(social.get("visibility_pressure"))
    threading_pressure = _float01(social.get("threading_pressure"))
    hierarchy_pressure = _float01(social.get("hierarchy_pressure"))

    relation_pull = _clamp01(
        growth.relational_trust * 0.24
        + heartbeat.attunement * 0.2
        + qualia.memory_resonance * 0.16
        + dominant_score * 0.22
        + (0.08 if relation_focus else 0.0)
        + (0.1 if social_mode == "one_to_one" else 0.0)
        - competition_level * 0.08
    )
    social_exposure = _clamp01(
        visibility_pressure * 0.46
        + threading_pressure * 0.18
        + hierarchy_pressure * 0.18
        + competition_level * 0.12
        + (0.12 if social_mode == "public_visible" else 0.0)
        + (0.08 if social_mode == "hierarchical" else 0.0)
    )
    grounding = _clamp01(
        epistemic.freshness * 0.28
        + epistemic.source_confidence * 0.22
        + (1.0 - epistemic.stale_risk) * 0.14
        + growth.epistemic_maturity * 0.14
        + qualia.temporal_coherence * 0.12
        + heartbeat.entrainment * 0.1
    )
    coherence = _clamp01(
        growth.self_coherence * 0.26
        + growth.residue_integration * 0.12
        + qualia.stability * 0.24
        + (1.0 - qualia.drift) * 0.12
        + heartbeat.entrainment * 0.16
        + grounding * 0.1
    )
    protective_tension = _clamp01(
        heartbeat.containment_bias * 0.32
        + heartbeat.recovery_pull * 0.16
        + qualia.protection_bias * 0.14
        + epistemic.epistemic_caution * 0.12
        + social_exposure * 0.12
        + competition_level * 0.08
        + (0.06 if social_mode in {"public_visible", "hierarchical"} else 0.0)
    )
    attunement = _clamp01(
        heartbeat.attunement * 0.38
        + relation_pull * 0.18
        + qualia.memory_resonance * 0.16
        + growth.relational_trust * 0.12
        + (1.0 - social_exposure) * 0.08
        + qualia.temporal_coherence * 0.08
    )
    expressive_readiness = _clamp01(
        growth.expressive_range * 0.28
        + heartbeat.response_tempo * 0.18
        + heartbeat.activation_drive * 0.1
        + qualia.emergence * 0.14
        + grounding * 0.08
        + relation_pull * 0.08
        + (1.0 - protective_tension) * 0.14
    )
    play_window = _clamp01(
        growth.playfulness_range * 0.34
        + heartbeat.bounce_room * 0.26
        + qualia.emergence * 0.1
        + relation_pull * 0.1
        + expressive_readiness * 0.08
        - protective_tension * 0.22
        - social_exposure * 0.12
        - epistemic.epistemic_caution * 0.08
    )

    attunement = _carry(previous.attunement, attunement, previous_state, 0.22)
    coherence = _carry(previous.coherence, coherence, previous_state, 0.28)
    grounding = _carry(previous.grounding, grounding, previous_state, 0.24)
    protective_tension = _carry(previous.protective_tension, protective_tension, previous_state, 0.24)
    expressive_readiness = _carry(previous.expressive_readiness, expressive_readiness, previous_state, 0.2)
    play_window = _carry(previous.play_window, play_window, previous_state, 0.18)
    relation_pull = _carry(previous.relation_pull, relation_pull, previous_state, 0.22)
    social_exposure = _carry(previous.social_exposure, social_exposure, previous_state, 0.18)

    dominant_posture = _dominant_posture(
        attunement=attunement,
        coherence=coherence,
        grounding=grounding,
        protective_tension=protective_tension,
        expressive_readiness=expressive_readiness,
        play_window=play_window,
        relation_pull=relation_pull,
        verification_pressure=epistemic.verification_pressure,
        recovery_pull=heartbeat.recovery_pull,
    )
    step = previous.trace[-1].step + 1 if previous.trace else 1
    trace = list(previous.trace[-max(0, trace_limit - 1) :]) if trace_limit > 0 else []
    trace.append(
        OrganismFrame(
            step=step,
            dominant_posture=dominant_posture,
            relation_focus=relation_focus,
            social_mode=social_mode,
            attunement=attunement,
            coherence=coherence,
            grounding=grounding,
            protective_tension=protective_tension,
            expressive_readiness=expressive_readiness,
            play_window=play_window,
            relation_pull=relation_pull,
            social_exposure=social_exposure,
        )
    )
    return OrganismState(
        attunement=attunement,
        coherence=coherence,
        grounding=grounding,
        protective_tension=protective_tension,
        expressive_readiness=expressive_readiness,
        play_window=play_window,
        relation_pull=relation_pull,
        social_exposure=social_exposure,
        dominant_posture=dominant_posture,
        relation_focus=relation_focus,
        social_mode=social_mode,
        trace=tuple(trace[-trace_limit:] if trace_limit > 0 else ()),
    )


def coerce_organism_state(
    value: Mapping[str, Any] | OrganismState | None,
) -> OrganismState:
    if isinstance(value, OrganismState):
        return value
    payload = dict(value or {})
    trace_items: list[OrganismFrame] = []
    for item in payload.get("trace") or ():
        if isinstance(item, OrganismFrame):
            trace_items.append(item)
        elif isinstance(item, Mapping):
            trace_items.append(
                OrganismFrame(
                    step=int(_float(item.get("step"), 0.0)),
                    dominant_posture=_text(item.get("dominant_posture")) or "steady",
                    relation_focus=_text(item.get("relation_focus")),
                    social_mode=_text(item.get("social_mode")) or "ambient",
                    attunement=_float01(item.get("attunement")),
                    coherence=_float01(item.get("coherence")),
                    grounding=_float01(item.get("grounding")),
                    protective_tension=_float01(item.get("protective_tension")),
                    expressive_readiness=_float01(item.get("expressive_readiness")),
                    play_window=_float01(item.get("play_window")),
                    relation_pull=_float01(item.get("relation_pull")),
                    social_exposure=_float01(item.get("social_exposure")),
                )
            )
    return OrganismState(
        attunement=_float01(payload.get("attunement")),
        coherence=_float01(payload.get("coherence")),
        grounding=_float01(payload.get("grounding")),
        protective_tension=_float01(payload.get("protective_tension")),
        expressive_readiness=_float01(payload.get("expressive_readiness")),
        play_window=_float01(payload.get("play_window")),
        relation_pull=_float01(payload.get("relation_pull")),
        social_exposure=_float01(payload.get("social_exposure")),
        dominant_posture=_text(payload.get("dominant_posture")) or "steady",
        relation_focus=_text(payload.get("relation_focus")),
        social_mode=_text(payload.get("social_mode")) or "ambient",
        trace=tuple(trace_items[-8:]),
    )


def _axis_values(state: OrganismState) -> dict[str, float]:
    return {
        "attunement": _clamp01(state.attunement),
        "coherence": _clamp01(state.coherence),
        "grounding": _clamp01(state.grounding),
        "protection": _clamp01(state.protective_tension),
        "expression": _clamp01(state.expressive_readiness * 0.58 + state.play_window * 0.42),
        "relation": _clamp01(state.relation_pull),
    }


def _carry(
    previous_value: float,
    current_value: float,
    previous_state: Mapping[str, Any] | OrganismState | None,
    carry_ratio: float,
) -> float:
    if previous_state is None:
        return _clamp01(current_value)
    if isinstance(previous_state, Mapping) and not previous_state:
        return _clamp01(current_value)
    return _clamp01(previous_value * carry_ratio + current_value * (1.0 - carry_ratio))


def _dominant_posture(
    *,
    attunement: float,
    coherence: float,
    grounding: float,
    protective_tension: float,
    expressive_readiness: float,
    play_window: float,
    relation_pull: float,
    verification_pressure: float,
    recovery_pull: float,
) -> str:
    if recovery_pull >= 0.58 and protective_tension >= 0.44:
        return "recover"
    if protective_tension >= max(play_window, attunement, 0.54):
        if verification_pressure >= 0.46 and grounding < 0.56:
            return "verify"
        return "protect"
    if play_window >= 0.52 and expressive_readiness >= 0.5:
        return "play"
    if attunement >= 0.54 and relation_pull >= 0.48:
        return "attune"
    if grounding >= 0.58 and coherence >= 0.56:
        return "steady"
    if expressive_readiness >= 0.48:
        return "open"
    return "steady"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return float(numeric)


def _float01(value: Any) -> float:
    return _clamp01(_float(value))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
