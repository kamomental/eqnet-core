from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .growth_state import coerce_growth_state
from .qualia_structure_state import coerce_qualia_structure_state


@dataclass(frozen=True)
class HeartbeatStructureFrame:
    rate_norm: float = 0.0
    phase: float = 0.0
    pulse_band: str = "soft_pulse"
    phase_window: str = "downbeat"
    dominant_reaction: str = "steady"
    activation_drive: float = 0.0
    containment_bias: float = 0.0
    bounce_room: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rate_norm": round(self.rate_norm, 4),
            "phase": round(self.phase, 4),
            "pulse_band": self.pulse_band,
            "phase_window": self.phase_window,
            "dominant_reaction": self.dominant_reaction,
            "activation_drive": round(self.activation_drive, 4),
            "containment_bias": round(self.containment_bias, 4),
            "bounce_room": round(self.bounce_room, 4),
        }


@dataclass(frozen=True)
class HeartbeatStructureState:
    rate: float = 0.85
    rate_norm: float = 0.2833
    phase: float = 0.0
    pulse_band: str = "soft_pulse"
    phase_window: str = "downbeat"
    activation_drive: float = 0.0
    attunement: float = 0.0
    containment_bias: float = 0.0
    recovery_pull: float = 0.0
    bounce_room: float = 0.0
    response_tempo: float = 0.0
    entrainment: float = 0.0
    dominant_reaction: str = "steady"
    trace: tuple[HeartbeatStructureFrame, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "rate": round(self.rate, 4),
            "rate_norm": round(self.rate_norm, 4),
            "phase": round(self.phase, 4),
            "pulse_band": self.pulse_band,
            "phase_window": self.phase_window,
            "activation_drive": round(self.activation_drive, 4),
            "attunement": round(self.attunement, 4),
            "containment_bias": round(self.containment_bias, 4),
            "recovery_pull": round(self.recovery_pull, 4),
            "bounce_room": round(self.bounce_room, 4),
            "response_tempo": round(self.response_tempo, 4),
            "entrainment": round(self.entrainment, 4),
            "dominant_reaction": self.dominant_reaction,
            "trace": [frame.to_dict() for frame in self.trace],
        }

    def to_packet_axes(
        self,
        previous: Mapping[str, Any] | "HeartbeatStructureState" | None = None,
    ) -> dict[str, dict[str, float]]:
        previous_state = coerce_heartbeat_structure_state(previous)
        current_axes = _axis_values(self)
        previous_axes = _axis_values(previous_state)
        return {
            axis_name: {
                "value": round(axis_value, 4),
                "delta": round(axis_value - previous_axes.get(axis_name, 0.0), 4),
            }
            for axis_name, axis_value in current_axes.items()
        }


def derive_heartbeat_structure_state(
    *,
    previous_state: Mapping[str, Any] | HeartbeatStructureState | None = None,
    heart_snapshot: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    qualia_structure_state: Mapping[str, Any] | None = None,
    qualia_planner_view: Mapping[str, Any] | None = None,
    growth_state: Mapping[str, Any] | None = None,
) -> HeartbeatStructureState:
    previous = coerce_heartbeat_structure_state(previous_state)
    heart = dict(heart_snapshot or {})
    metric_payload = dict(metrics or {})
    planner = dict(qualia_planner_view or {})
    qualia = coerce_qualia_structure_state(qualia_structure_state)
    growth = coerce_growth_state(growth_state)

    rate = _clamp(float(heart.get("rate") or previous.rate or 0.85), 0.2, 3.0)
    rate_norm = _clamp01(float(metric_payload.get("heart_rate_norm") or rate / 3.0))
    phase = _phase01(heart.get("phase", previous.phase))
    life_indicator = _clamp01(float(metric_payload.get("life_indicator") or 0.0))
    tension = _clamp01(float(metric_payload.get("tension_score") or 0.0))
    phi_norm = _clamp01(float(metric_payload.get("phi_norm") or 0.0))
    trust = _clamp01(float(planner.get("trust") or growth.relational_trust))
    body_load = _clamp01(float(planner.get("body_load") or 0.0))
    protection_bias = _clamp01(float(planner.get("protection_bias") or 0.0))

    activation_drive = _clamp01(
        rate_norm * 0.34
        + life_indicator * 0.22
        + qualia.emergence * 0.24
        + phi_norm * 0.2
    )
    containment_bias = _clamp01(
        tension * 0.24
        + protection_bias * 0.28
        + body_load * 0.18
        + max(0.0, 0.55 - trust) * 0.24
        + (1.0 - qualia.stability) * 0.16
        + qualia.drift * 0.14
    )
    recovery_pull = _clamp01(
        body_load * 0.38
        + containment_bias * 0.22
        + (1.0 - life_indicator) * 0.18
        + (1.0 - qualia.stability) * 0.12
        + protection_bias * 0.1
    )
    attunement = _clamp01(
        life_indicator * 0.22
        + qualia.memory_resonance * 0.28
        + qualia.temporal_coherence * 0.18
        + growth.relational_trust * 0.2
        + trust * 0.12
        - containment_bias * 0.12
    )
    bounce_room = _clamp01(
        growth.playfulness_range * 0.35
        + growth.expressive_range * 0.2
        + rate_norm * 0.14
        + life_indicator * 0.12
        + qualia.emergence * 0.12
        - containment_bias * 0.3
        - recovery_pull * 0.1
    )
    phase_alignment = 1.0 - min(abs(phase - previous.phase), 1.0)
    entrainment = _clamp01(
        qualia.temporal_coherence * 0.32
        + qualia.stability * 0.18
        + attunement * 0.26
        + (1.0 - min(abs(rate_norm - previous.rate_norm), 1.0)) * 0.16
        + phase_alignment * 0.08
    )
    response_tempo = _clamp01(
        rate_norm * 0.34
        + bounce_room * 0.28
        + activation_drive * 0.12
        + attunement * 0.1
        - containment_bias * 0.18
        - recovery_pull * 0.16
    )
    pulse_band = _pulse_band(rate_norm)
    phase_window = _phase_window(phase)
    dominant_reaction = _dominant_reaction(
        containment_bias=containment_bias,
        recovery_pull=recovery_pull,
        bounce_room=bounce_room,
        attunement=attunement,
        activation_drive=activation_drive,
    )
    frame = HeartbeatStructureFrame(
        rate_norm=rate_norm,
        phase=phase,
        pulse_band=pulse_band,
        phase_window=phase_window,
        dominant_reaction=dominant_reaction,
        activation_drive=activation_drive,
        containment_bias=containment_bias,
        bounce_room=bounce_room,
    )
    trace = _next_trace(previous.trace, frame)
    return HeartbeatStructureState(
        rate=rate,
        rate_norm=rate_norm,
        phase=phase,
        pulse_band=pulse_band,
        phase_window=phase_window,
        activation_drive=activation_drive,
        attunement=attunement,
        containment_bias=containment_bias,
        recovery_pull=recovery_pull,
        bounce_room=bounce_room,
        response_tempo=response_tempo,
        entrainment=entrainment,
        dominant_reaction=dominant_reaction,
        trace=trace,
    )


def coerce_heartbeat_structure_state(
    value: Mapping[str, Any] | HeartbeatStructureState | None,
) -> HeartbeatStructureState:
    if isinstance(value, HeartbeatStructureState):
        return value
    payload = dict(value or {})
    raw_trace = payload.get("trace") or []
    trace: list[HeartbeatStructureFrame] = []
    if isinstance(raw_trace, (list, tuple)):
        for item in raw_trace:
            if isinstance(item, Mapping):
                trace.append(
                    HeartbeatStructureFrame(
                        rate_norm=_clamp01(float(item.get("rate_norm") or 0.0)),
                        phase=_phase01(item.get("phase", 0.0)),
                        pulse_band=str(item.get("pulse_band") or "soft_pulse").strip() or "soft_pulse",
                        phase_window=str(item.get("phase_window") or "downbeat").strip() or "downbeat",
                        dominant_reaction=str(item.get("dominant_reaction") or "steady").strip() or "steady",
                        activation_drive=_clamp01(float(item.get("activation_drive") or 0.0)),
                        containment_bias=_clamp01(float(item.get("containment_bias") or 0.0)),
                        bounce_room=_clamp01(float(item.get("bounce_room") or 0.0)),
                    )
                )
    return HeartbeatStructureState(
        rate=_clamp(float(payload.get("rate") or 0.85), 0.2, 3.0),
        rate_norm=_clamp01(float(payload.get("rate_norm") or 0.2833)),
        phase=_phase01(payload.get("phase", 0.0)),
        pulse_band=str(payload.get("pulse_band") or "soft_pulse").strip() or "soft_pulse",
        phase_window=str(payload.get("phase_window") or "downbeat").strip() or "downbeat",
        activation_drive=_clamp01(float(payload.get("activation_drive") or 0.0)),
        attunement=_clamp01(float(payload.get("attunement") or 0.0)),
        containment_bias=_clamp01(float(payload.get("containment_bias") or 0.0)),
        recovery_pull=_clamp01(float(payload.get("recovery_pull") or 0.0)),
        bounce_room=_clamp01(float(payload.get("bounce_room") or 0.0)),
        response_tempo=_clamp01(float(payload.get("response_tempo") or 0.0)),
        entrainment=_clamp01(float(payload.get("entrainment") or 0.0)),
        dominant_reaction=str(payload.get("dominant_reaction") or "steady").strip() or "steady",
        trace=tuple(trace[-8:]),
    )


def _axis_values(state: HeartbeatStructureState) -> dict[str, float]:
    return {
        "activation": _clamp01(state.activation_drive),
        "attunement": _clamp01(state.attunement),
        "containment": _clamp01(state.containment_bias),
        "recovery": _clamp01(state.recovery_pull),
        "tempo": _clamp01(state.response_tempo),
    }


def _pulse_band(rate_norm: float) -> str:
    if rate_norm < 0.34:
        return "soft_pulse"
    if rate_norm < 0.62:
        return "lifted_pulse"
    return "racing_pulse"


def _phase_window(phase: float) -> str:
    if phase < 0.25:
        return "downbeat"
    if phase < 0.5:
        return "upswing"
    if phase < 0.75:
        return "crest"
    return "release"


def _dominant_reaction(
    *,
    containment_bias: float,
    recovery_pull: float,
    bounce_room: float,
    attunement: float,
    activation_drive: float,
) -> str:
    if recovery_pull >= max(bounce_room, attunement, 0.58):
        return "recover"
    if containment_bias >= max(bounce_room, attunement, 0.56):
        return "contain"
    if bounce_room >= 0.44 and activation_drive >= 0.38:
        return "bounce"
    if attunement >= 0.34:
        return "attune"
    return "steady"


def _next_trace(
    previous_trace: tuple[HeartbeatStructureFrame, ...],
    frame: HeartbeatStructureFrame,
) -> tuple[HeartbeatStructureFrame, ...]:
    trace = list(previous_trace)
    trace.append(frame)
    return tuple(trace[-8:])


def _phase01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    numeric = numeric % 1.0
    return _clamp01(numeric)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))
