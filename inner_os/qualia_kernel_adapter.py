from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .observation_model import Observation, TensorObservationModel, Vector
from .qualia_projector import BasicQualiaProjector, QualiaProjector, QualiaState
from .self_estimator import (
    Estimate,
    EstimatorHealth,
    ResidualLinearSelfEstimator,
    evaluate_estimator_health,
)

QUALIA_EXTERNAL_AXES = (
    "mutual_attention_signal",
    "repair_signal",
    "hesitation_signal",
    "proximity_delta",
)
QUALIA_BODY_AXES = (
    "stress_level",
    "recovery_need",
    "voice_level",
    "autonomic_balance",
)
QUALIA_BOUNDARY_AXES = (
    "safety_bias",
    "near_body_risk",
    "defensive_level",
    "caution_bias",
)
QUALIA_ACTION_AXES = (
    "surface_policy_active",
    "approach_confidence",
)
QUALIA_AXIS_LABELS = (
    *QUALIA_EXTERNAL_AXES,
    *QUALIA_BODY_AXES,
    *QUALIA_BOUNDARY_AXES,
    *QUALIA_ACTION_AXES,
)
QUALIA_STATE_DIM = len(QUALIA_AXIS_LABELS)


@dataclass(frozen=True)
class QualiaPlannerView:
    trust: float
    degraded: bool
    dominant_axis: str | None
    dominant_value: float
    body_load: float
    protection_bias: float
    felt_energy: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trust": float(self.trust),
            "degraded": bool(self.degraded),
            "dominant_axis": self.dominant_axis,
            "dominant_value": float(self.dominant_value),
            "body_load": float(self.body_load),
            "protection_bias": float(self.protection_bias),
            "felt_energy": float(self.felt_energy),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "QualiaPlannerView":
        source = dict(payload or {})
        dominant_axis = source.get("dominant_axis")
        return cls(
            trust=float(source.get("trust", 1.0) or 0.0),
            degraded=bool(source.get("degraded", False)),
            dominant_axis=str(dominant_axis) if dominant_axis else None,
            dominant_value=float(source.get("dominant_value", 0.0) or 0.0),
            body_load=float(source.get("body_load", 0.0) or 0.0),
            protection_bias=float(source.get("protection_bias", 0.0) or 0.0),
            felt_energy=float(source.get("felt_energy", 0.0) or 0.0),
        )


@dataclass(frozen=True)
class RuntimeQualiaKernelState:
    observation: Observation
    estimate: Estimate
    health: EstimatorHealth
    qualia_state: QualiaState
    protection_grad_x: Vector
    axis_labels: tuple[str, ...]

    def to_hint_dict(self) -> Dict[str, Any]:
        return {
            "axis_labels": list(self.axis_labels),
            "health": self.health.to_dict(),
            "state": self.qualia_state.to_dict(),
            "protection_grad_x": self.protection_grad_x.astype(np.float32).tolist(),
        }


def build_default_runtime_observation_model() -> TensorObservationModel:
    return TensorObservationModel(
        latent_dim=QUALIA_STATE_DIM,
        ext_size=len(QUALIA_EXTERNAL_AXES),
        body_size=len(QUALIA_BODY_AXES),
        boundary_size=len(QUALIA_BOUNDARY_AXES),
        action_size=len(QUALIA_ACTION_AXES),
    )


def build_qualia_planner_view(
    *,
    qualia_state: Mapping[str, Any] | QualiaState | None,
    estimator_health: Mapping[str, Any] | EstimatorHealth | None = None,
    protection_grad_x: Sequence[float] | None = None,
    axis_labels: Sequence[str] | None = None,
) -> QualiaPlannerView:
    if qualia_state is None:
        return QualiaPlannerView(
            trust=1.0,
            degraded=False,
            dominant_axis=None,
            dominant_value=0.0,
            body_load=0.0,
            protection_bias=0.0,
            felt_energy=0.0,
        )

    state_dict = qualia_state.to_dict() if isinstance(qualia_state, QualiaState) else dict(qualia_state)
    trust = float(
        (
            estimator_health.trust
            if isinstance(estimator_health, EstimatorHealth)
            else (estimator_health or {}).get("trust", state_dict.get("trust_applied", 1.0))
        )
        or 0.0
    )
    degraded = bool(
        (
            estimator_health.degraded
            if isinstance(estimator_health, EstimatorHealth)
            else (estimator_health or {}).get("degraded", state_dict.get("degraded", False))
        )
    )
    q = _coerce_view_vector(state_dict.get("qualia"), axis_labels)
    gate = _coerce_view_vector(state_dict.get("gate"), axis_labels)
    body_coupling = _coerce_view_vector(state_dict.get("body_coupling"), axis_labels)
    value_grad = _coerce_view_vector(state_dict.get("value_grad"), axis_labels)
    labels = tuple(str(item) for item in (axis_labels or state_dict.get("axis_labels") or QUALIA_AXIS_LABELS))
    if len(labels) < q.size:
        labels = tuple(list(labels) + [f"axis_{idx}" for idx in range(len(labels), q.size)])
    elif len(labels) > q.size:
        labels = labels[: q.size]

    weighted = np.abs(q) * np.maximum(gate, 0.0)
    dominant_index = int(np.argmax(weighted)) if weighted.size and float(np.max(weighted)) > 1.0e-6 else -1
    dominant_axis = labels[dominant_index] if dominant_index >= 0 else None
    dominant_value = float(weighted[dominant_index]) if dominant_index >= 0 else 0.0
    felt_energy = float(np.mean(np.abs(q))) if q.size else 0.0
    body_load = float(np.mean(np.abs(q) * np.maximum(body_coupling, 0.0))) if q.size else 0.0
    protection_seed = _coerce_view_vector(protection_grad_x, labels) if protection_grad_x is not None else value_grad
    protection_bias = float(
        np.mean(np.abs(q) * np.maximum(value_grad + np.abs(protection_seed), 0.0))
    ) if q.size else 0.0
    return QualiaPlannerView(
        trust=max(0.0, min(1.0, trust)),
        degraded=degraded,
        dominant_axis=dominant_axis,
        dominant_value=max(0.0, dominant_value),
        body_load=max(0.0, body_load),
        protection_bias=max(0.0, protection_bias),
        felt_energy=max(0.0, felt_energy),
    )


@dataclass
class RuntimeQualiaKernelAdapter:
    observation_model: TensorObservationModel = field(
        default_factory=build_default_runtime_observation_model
    )
    projector: QualiaProjector = field(default_factory=BasicQualiaProjector)
    default_dt: float = 1.0

    def step(
        self,
        *,
        current_state: Mapping[str, Any],
        safety_signals: Mapping[str, Any],
        prev_qualia: Sequence[float],
        prev_habituation: Sequence[float],
        prev_protection_grad_x: Sequence[float],
        dt: float | None = None,
    ) -> RuntimeQualiaKernelState:
        step_dt = float(dt if dt is not None else self.default_dt)
        if step_dt <= 0.0:
            raise ValueError("qualia kernel dt must be positive")

        raw_sensors = _build_raw_sensors(current_state, safety_signals)
        zero_body = np.zeros(len(QUALIA_BODY_AXES), dtype=np.float32)
        zero_action = np.zeros(len(QUALIA_ACTION_AXES), dtype=np.float32)
        observation = self.observation_model.encode(
            raw_sensors=raw_sensors,
            body=zero_body,
            prev_action=zero_action,
            dt=step_dt,
        )
        prior_state = _build_prior_state_vector(current_state)
        estimator = ResidualLinearSelfEstimator(
            observation_model=self.observation_model,
            latent_dim=QUALIA_STATE_DIM,
            initial_state=prior_state,
        )
        estimate = estimator.step(observation, zero_body, zero_action)
        health = evaluate_estimator_health(estimate, observation)
        memory = _build_memory_vector(current_state)
        qualia_state = self.projector.project(
            obs=observation,
            est=estimate,
            health=health,
            memory=memory,
            prev_qualia=_coerce_vector(prev_qualia),
            prev_habituation=_coerce_vector(prev_habituation),
            protection_grad_x=_coerce_vector(prev_protection_grad_x),
            dt=step_dt,
        )
        protection_grad_x = _build_protection_grad_x(current_state, safety_signals)
        return RuntimeQualiaKernelState(
            observation=observation,
            estimate=estimate,
            health=health,
            qualia_state=qualia_state,
            protection_grad_x=protection_grad_x,
            axis_labels=tuple(QUALIA_AXIS_LABELS),
        )


def _build_raw_sensors(
    current_state: Mapping[str, Any],
    safety_signals: Mapping[str, Any],
) -> Dict[str, list[float | None]]:
    return {
        "ext": [
            _maybe_float(safety_signals.get("mutual_attention_score")),
            _maybe_float(safety_signals.get("repair_signal")),
            _maybe_float(safety_signals.get("hesitation_signal")),
            _maybe_float(safety_signals.get("proximity_delta")),
        ],
        "body_obs": [
            _float_from(current_state, "stress", default=0.0),
            _float_from(current_state, "recovery_need", default=0.0),
            _float_from(current_state, "voice_level", default=0.0),
            _float_from(current_state, "autonomic_balance", default=0.5),
        ],
        "boundary": [
            max(
                _float_from(current_state, "safety_bias", default=0.0),
                _float_from(safety_signals, "safety_bias", default=0.0),
            ),
            _float_from(current_state, "near_body_risk", default=0.0),
            _float_from(current_state, "defensive_level", default=0.0),
            _float_from(current_state, "caution_bias", default=0.0),
        ],
        "action_feedback": [
            _float_from(current_state, "surface_policy_active", default=0.0),
            _float_from(current_state, "approach_confidence", default=0.0),
        ],
    }


def _build_prior_state_vector(current_state: Mapping[str, Any]) -> Vector:
    shared_attention = _float_from(current_state, "predicted_shared_attention", default=0.0)
    if shared_attention <= 0.0:
        shared_attention = _float_from(current_state, "social_grounding", default=0.0)
    return np.asarray(
        [
            shared_attention,
            _float_from(current_state, "stream_repair_window_hold", default=0.0),
            _float_from(current_state, "recent_strain", default=0.0),
            _float_from(current_state, "reachability", default=0.0),
            _float_from(current_state, "stress", default=0.0),
            _float_from(current_state, "recovery_need", default=0.0),
            _float_from(current_state, "voice_level", default=0.0),
            _float_from(current_state, "autonomic_balance", default=0.5),
            _float_from(current_state, "safety_bias", default=0.0),
            _float_from(current_state, "near_body_risk", default=0.0),
            _float_from(current_state, "defensive_level", default=0.0),
            _float_from(current_state, "caution_bias", default=0.0),
            _float_from(current_state, "surface_policy_active", default=0.0),
            _float_from(current_state, "approach_confidence", default=0.0),
        ],
        dtype=np.float32,
    )


def _build_memory_vector(current_state: Mapping[str, Any]) -> Vector:
    return np.asarray(
        [
            _float_from(current_state, "interaction_afterglow", default=0.0),
            _float_from(current_state, "replay_intensity", default=0.0),
            _float_from(current_state, "meaning_inertia", default=0.0),
            _float_from(current_state, "conscious_residue_strength", default=0.0),
        ],
        dtype=np.float32,
    )


def _build_protection_grad_x(
    current_state: Mapping[str, Any],
    safety_signals: Mapping[str, Any],
) -> Vector:
    predicted_shared_attention = _float_from(current_state, "predicted_shared_attention", default=0.0)
    if predicted_shared_attention <= 0.0:
        predicted_shared_attention = _float_from(current_state, "social_grounding", default=0.0)
    mutual_attention_score = _maybe_float(safety_signals.get("mutual_attention_score"))
    mutual_attention_gap = 1.0 - _clamp01(
        mutual_attention_score
        if mutual_attention_score is not None
        else predicted_shared_attention
    )
    return np.asarray(
        [
            mutual_attention_gap,
            max(
                _float_from(current_state, "stream_repair_window_hold", default=0.0),
                _float_from(safety_signals, "repair_signal", default=0.0),
            ),
            _float_from(current_state, "recent_strain", default=0.0),
            1.0 - _clamp01(_float_from(current_state, "reachability", default=0.0)),
            _float_from(current_state, "stress", default=0.0),
            _float_from(current_state, "recovery_need", default=0.0),
            abs(_float_from(current_state, "voice_level", default=0.0)),
            abs(_float_from(current_state, "autonomic_balance", default=0.5) - 0.5) * 2.0,
            max(
                _float_from(current_state, "safety_bias", default=0.0),
                _float_from(safety_signals, "safety_bias", default=0.0),
            ),
            _float_from(current_state, "near_body_risk", default=0.0),
            _float_from(current_state, "defensive_level", default=0.0),
            _float_from(current_state, "caution_bias", default=0.0),
            _float_from(current_state, "surface_policy_active", default=0.0),
            1.0 - _clamp01(_float_from(current_state, "approach_confidence", default=0.0)),
        ],
        dtype=np.float32,
    )


def _coerce_vector(values: Sequence[float]) -> Vector:
    vector = np.asarray(list(values), dtype=np.float32).reshape(-1)
    if vector.size < QUALIA_STATE_DIM:
        vector = np.concatenate(
            [vector, np.zeros(QUALIA_STATE_DIM - vector.size, dtype=np.float32)]
        )
    elif vector.size > QUALIA_STATE_DIM:
        vector = vector[:QUALIA_STATE_DIM]
    return vector.astype(np.float32)


def _coerce_view_vector(values: Sequence[float] | None, axis_labels: Sequence[str] | None) -> Vector:
    target_size = len(axis_labels or QUALIA_AXIS_LABELS)
    if values is None:
        return np.zeros(target_size, dtype=np.float32)
    vector = np.asarray(list(values), dtype=np.float32).reshape(-1)
    if vector.size < target_size:
        vector = np.concatenate(
            [vector, np.zeros(target_size - vector.size, dtype=np.float32)]
        )
    elif vector.size > target_size:
        vector = vector[:target_size]
    return vector.astype(np.float32)


def _float_from(mapping: Mapping[str, Any], key: str, *, default: float = 0.0) -> float:
    try:
        return float(mapping.get(key, default) or 0.0)
    except (TypeError, ValueError):
        return float(default)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
