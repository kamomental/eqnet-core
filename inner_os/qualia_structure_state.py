from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class QualiaStructureFrame:
    """クオリア構造の短期フレーム。"""

    step: int = 0
    dominant_axis: str = ""
    phase: str = "ambient"
    intensity: float = 0.0
    emergence: float = 0.0
    stability: float = 0.0
    memory_resonance: float = 0.0
    temporal_coherence: float = 0.0
    drift: float = 0.0
    trust: float = 0.0
    body_load: float = 0.0
    protection_bias: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "dominant_axis": self.dominant_axis,
            "phase": self.phase,
            "intensity": round(self.intensity, 4),
            "emergence": round(self.emergence, 4),
            "stability": round(self.stability, 4),
            "memory_resonance": round(self.memory_resonance, 4),
            "temporal_coherence": round(self.temporal_coherence, 4),
            "drift": round(self.drift, 4),
            "trust": round(self.trust, 4),
            "body_load": round(self.body_load, 4),
            "protection_bias": round(self.protection_bias, 4),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "QualiaStructureFrame":
        return cls(
            step=int(_float(payload.get("step"), 0.0)),
            dominant_axis=_text(payload.get("dominant_axis")),
            phase=_text(payload.get("phase")) or "ambient",
            intensity=_float01(payload.get("intensity")),
            emergence=_float01(payload.get("emergence")),
            stability=_float01(payload.get("stability")),
            memory_resonance=_float01(payload.get("memory_resonance")),
            temporal_coherence=_float01(payload.get("temporal_coherence")),
            drift=_float01(payload.get("drift")),
            trust=_float01(payload.get("trust")),
            body_load=_float01(payload.get("body_load")),
            protection_bias=_float01(payload.get("protection_bias")),
        )


@dataclass(frozen=True)
class QualiaStructureState:
    """クオリア構造の内部時系列状態。"""

    center: tuple[float, ...] = ()
    momentum: tuple[float, ...] = ()
    axis_labels: tuple[str, ...] = ()
    dominant_axis: str = ""
    phase: str = "ambient"
    intensity: float = 0.0
    emergence: float = 0.0
    stability: float = 0.0
    memory_resonance: float = 0.0
    temporal_coherence: float = 0.0
    drift: float = 0.0
    trust: float = 0.0
    body_load: float = 0.0
    protection_bias: float = 0.0
    degraded: bool = False
    trace: tuple[QualiaStructureFrame, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "center": [round(float(value), 4) for value in self.center],
            "momentum": [round(float(value), 4) for value in self.momentum],
            "axis_labels": list(self.axis_labels),
            "dominant_axis": self.dominant_axis,
            "phase": self.phase,
            "intensity": round(self.intensity, 4),
            "emergence": round(self.emergence, 4),
            "stability": round(self.stability, 4),
            "memory_resonance": round(self.memory_resonance, 4),
            "temporal_coherence": round(self.temporal_coherence, 4),
            "drift": round(self.drift, 4),
            "trust": round(self.trust, 4),
            "body_load": round(self.body_load, 4),
            "protection_bias": round(self.protection_bias, 4),
            "degraded": bool(self.degraded),
            "trace": [frame.to_dict() for frame in self.trace],
        }

    def to_packet_axes(
        self,
        previous: Mapping[str, Any] | "QualiaStructureState" | None = None,
    ) -> dict[str, dict[str, float]]:
        previous_state = coerce_qualia_structure_state(previous)
        current_axes = _axis_values(self)
        previous_axes = _axis_values(previous_state)
        return {
            axis_name: {
                "value": round(axis_value, 4),
                "delta": round(axis_value - previous_axes.get(axis_name, 0.0), 4),
            }
            for axis_name, axis_value in current_axes.items()
        }


def derive_qualia_structure_state(
    *,
    previous_state: Mapping[str, Any] | QualiaStructureState | None,
    qualia_state: Mapping[str, Any] | None,
    temporal_membrane_bias: Mapping[str, Any] | None,
    qualia_planner_view: Mapping[str, Any] | None,
    trace_limit: int = 8,
) -> QualiaStructureState:
    previous = coerce_qualia_structure_state(previous_state)
    qualia_payload = dict(qualia_state or {})
    temporal_payload = dict(temporal_membrane_bias or {})
    planner_payload = dict(qualia_planner_view or {})

    center_vec = np.asarray(_float_sequence(qualia_payload.get("qualia")), dtype=np.float32)
    if center_vec.size <= 0:
        return previous

    axis_labels = tuple(
        _text(item)
        for item in qualia_payload.get("axis_labels") or ()
        if _text(item)
    )
    gate_vec = _coerce_like(center_vec, qualia_payload.get("gate"))
    habituation_vec = _coerce_like(center_vec, qualia_payload.get("habituation"))
    body_vec = _coerce_like(center_vec, qualia_payload.get("body_coupling"))

    prev_center = np.asarray(previous.center, dtype=np.float32)
    if prev_center.shape != center_vec.shape:
        prev_center = np.zeros_like(center_vec, dtype=np.float32)
    momentum_vec = center_vec - prev_center

    trust = _float01(planner_payload.get("trust", qualia_payload.get("trust_applied")))
    protection_bias = _float01(planner_payload.get("protection_bias"))
    body_load = max(
        _float01(planner_payload.get("body_load")),
        _float01(float(np.mean(np.maximum(body_vec, 0.0))) if body_vec.size else 0.0),
    )
    temporal_coherence = _float01(temporal_payload.get("timeline_coherence"))
    reentry_pull = _float01(temporal_payload.get("reentry_pull"))
    continuity_pressure = _float01(temporal_payload.get("continuity_pressure"))
    relation_reentry_pull = _float01(temporal_payload.get("relation_reentry_pull"))
    supersession_pressure = _float01(temporal_payload.get("supersession_pressure"))
    degraded = bool(
        planner_payload.get("degraded", qualia_payload.get("degraded", False))
    )

    weighted = np.abs(center_vec) * np.maximum(gate_vec, 0.0) * max(trust, 0.12)
    intensity = _vector_energy(weighted)
    drift = _vector_energy(momentum_vec)
    memory_resonance = _clamp01(
        temporal_coherence * 0.34
        + reentry_pull * 0.22
        + continuity_pressure * 0.28
        + relation_reentry_pull * 0.16
        - supersession_pressure * 0.08
    )
    habituation_load = _float01(float(np.mean(np.abs(habituation_vec))) if habituation_vec.size else 0.0)
    felt_energy = max(_float01(planner_payload.get("felt_energy")), intensity)
    stability = _clamp01(
        trust * 0.24
        + (1.0 - drift) * 0.22
        + habituation_load * 0.18
        + memory_resonance * 0.14
        + (1.0 - protection_bias) * 0.12
        + (0.0 if degraded else 0.1)
    )
    emergence = _clamp01(
        felt_energy * 0.3
        + drift * 0.24
        + intensity * 0.18
        + memory_resonance * 0.12
        + trust * 0.1
        - protection_bias * 0.1
        - supersession_pressure * 0.06
        - (0.12 if degraded else 0.0)
    )

    dominant_axis = _dominant_axis(axis_labels, weighted)
    phase = _derive_phase(
        emergence=emergence,
        stability=stability,
        memory_resonance=memory_resonance,
        drift=drift,
        trust=trust,
        degraded=degraded,
        supersession_pressure=supersession_pressure,
    )

    next_step = previous.trace[-1].step + 1 if previous.trace else 1
    trace = list(previous.trace[-max(0, trace_limit - 1) :]) if trace_limit > 0 else []
    trace.append(
        QualiaStructureFrame(
            step=next_step,
            dominant_axis=dominant_axis,
            phase=phase,
            intensity=intensity,
            emergence=emergence,
            stability=stability,
            memory_resonance=memory_resonance,
            temporal_coherence=temporal_coherence,
            drift=drift,
            trust=trust,
            body_load=body_load,
            protection_bias=protection_bias,
        )
    )

    return QualiaStructureState(
        center=tuple(float(value) for value in center_vec.tolist()),
        momentum=tuple(float(value) for value in momentum_vec.tolist()),
        axis_labels=axis_labels,
        dominant_axis=dominant_axis,
        phase=phase,
        intensity=intensity,
        emergence=emergence,
        stability=stability,
        memory_resonance=memory_resonance,
        temporal_coherence=temporal_coherence,
        drift=drift,
        trust=trust,
        body_load=body_load,
        protection_bias=protection_bias,
        degraded=degraded,
        trace=tuple(trace[-trace_limit:] if trace_limit > 0 else ()),
    )


def coerce_qualia_structure_state(
    value: Mapping[str, Any] | QualiaStructureState | None,
) -> QualiaStructureState:
    if isinstance(value, QualiaStructureState):
        return value
    payload = dict(value or {})
    trace_items = []
    for item in payload.get("trace") or ():
        if isinstance(item, QualiaStructureFrame):
            trace_items.append(item)
        elif isinstance(item, Mapping):
            trace_items.append(QualiaStructureFrame.from_mapping(item))
    return QualiaStructureState(
        center=_float_sequence(payload.get("center")),
        momentum=_float_sequence(payload.get("momentum")),
        axis_labels=tuple(_text(item) for item in payload.get("axis_labels") or () if _text(item)),
        dominant_axis=_text(payload.get("dominant_axis")),
        phase=_text(payload.get("phase")) or "ambient",
        intensity=_float01(payload.get("intensity")),
        emergence=_float01(payload.get("emergence")),
        stability=_float01(payload.get("stability")),
        memory_resonance=_float01(payload.get("memory_resonance")),
        temporal_coherence=_float01(payload.get("temporal_coherence")),
        drift=_float01(payload.get("drift")),
        trust=_float01(payload.get("trust")),
        body_load=_float01(payload.get("body_load")),
        protection_bias=_float01(payload.get("protection_bias")),
        degraded=bool(payload.get("degraded", False)),
        trace=tuple(trace_items),
    )


def _axis_values(state: QualiaStructureState) -> dict[str, float]:
    return {
        "emergence": _clamp01(
            state.emergence * 0.62
            + state.intensity * 0.24
            + state.trust * 0.14
        ),
        "stability": _clamp01(
            state.stability * 0.68
            + state.temporal_coherence * 0.18
            + (1.0 - state.drift) * 0.14
        ),
        "resonance": _clamp01(
            state.memory_resonance * 0.72
            + state.temporal_coherence * 0.18
            + state.trust * 0.1
        ),
        "drift": _clamp01(
            state.drift * 0.7
            + state.emergence * 0.18
            + (0.12 if state.phase in {"rising", "shifting"} else 0.0)
        ),
    }


def _coerce_like(reference: np.ndarray, values: Any) -> np.ndarray:
    vector = np.asarray(_float_sequence(values), dtype=np.float32)
    if vector.shape == reference.shape:
        return vector
    if vector.size <= 0:
        return np.zeros_like(reference, dtype=np.float32)
    resized = np.zeros_like(reference, dtype=np.float32)
    size = min(reference.size, vector.size)
    resized[:size] = vector[:size]
    return resized


def _dominant_axis(axis_labels: Sequence[str], weighted: np.ndarray) -> str:
    if weighted.size <= 0:
        return ""
    index = int(np.argmax(np.abs(weighted)))
    if axis_labels and index < len(axis_labels):
        return _text(axis_labels[index])
    return f"axis_{index}"


def _derive_phase(
    *,
    emergence: float,
    stability: float,
    memory_resonance: float,
    drift: float,
    trust: float,
    degraded: bool,
    supersession_pressure: float,
) -> str:
    if degraded or (supersession_pressure >= 0.48 and trust <= 0.45):
        return "fragmenting"
    if emergence >= 0.58 and drift >= 0.22:
        return "rising"
    if memory_resonance >= 0.48 and drift <= 0.18:
        return "echoing"
    if stability >= 0.56 and drift <= 0.16:
        return "settling"
    if drift >= 0.18:
        return "shifting"
    return "holding"


def _vector_energy(vector: np.ndarray) -> float:
    if vector.size <= 0:
        return 0.0
    return _clamp01(float(np.linalg.norm(vector)) / max(np.sqrt(float(vector.size)), 1.0))


def _float_sequence(values: Any) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    result: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        result.append(float(numeric))
    return tuple(result)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _float(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return float(numeric)


def _float01(value: Any) -> float:
    return _clamp01(_float(value, 0.0))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
