from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from .observation_model import Matrix, Observation, ObservationLayout, Vector
from .self_estimator import Estimate, EstimatorHealth


@dataclass(frozen=True)
class QualiaState:
    gate: Vector
    qualia: Vector
    precision: Vector
    observability: Vector
    body_coupling: Vector
    value_grad: Vector
    habituation: Vector
    trust_applied: float
    degraded: bool
    reason: str | None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate": self.gate.tolist(),
            "qualia": self.qualia.tolist(),
            "precision": self.precision.tolist(),
            "observability": self.observability.tolist(),
            "body_coupling": self.body_coupling.tolist(),
            "value_grad": self.value_grad.tolist(),
            "habituation": self.habituation.tolist(),
            "trust_applied": float(self.trust_applied),
            "degraded": bool(self.degraded),
            "reason": self.reason,
        }


class QualiaProjector(Protocol):
    def project(
        self,
        obs: Observation,
        est: Estimate,
        health: EstimatorHealth,
        memory: Sequence[float],
        prev_qualia: Sequence[float],
        prev_habituation: Sequence[float],
        protection_grad_x: Sequence[float],
        dt: float,
    ) -> QualiaState: ...


@dataclass
class BasicQualiaProjector:
    alpha_observability: float = 1.0
    alpha_precision: float = 1.0
    alpha_body: float = 0.8
    alpha_value: float = 0.8
    alpha_habituation: float = 0.7
    theta_q: float = 1.4
    habituation_decay: float = 0.86
    precision_floor: float = 1.0e-6
    projector_matrix: Matrix | None = None

    def project(
        self,
        obs: Observation,
        est: Estimate,
        health: EstimatorHealth,
        memory: Sequence[float],
        prev_qualia: Sequence[float],
        prev_habituation: Sequence[float],
        protection_grad_x: Sequence[float],
        dt: float,
    ) -> QualiaState:
        del memory
        if dt <= 0.0:
            raise ValueError("qualia projector dt must be positive")

        state_dim = est.x_hat.shape[0]
        prev_q = _as_vector(prev_qualia, state_dim, "prev_qualia")
        prev_h = _as_vector(prev_habituation, state_dim, "prev_habituation")
        value_grad = np.abs(_as_vector(protection_grad_x, state_dim, "protection_grad_x"))

        projector = self._projector(state_dim)
        precision = self._precision(est)
        observability = self._observability(obs, est)
        body_coupling = self._body_coupling(obs.layout, obs.mask, est.H, state_dim)
        habituation = self._habituation(prev_h, prev_q)

        gate_raw = (
            self.alpha_observability * _normalize(observability)
            + self.alpha_precision * _normalize(precision)
            + self.alpha_body * _normalize(body_coupling)
            + self.alpha_value * _normalize(value_grad)
            - self.alpha_habituation * _normalize(habituation)
            - self.theta_q
        )
        gate = _sigmoid(gate_raw).astype(np.float32)
        gate = (float(health.trust) * gate).astype(np.float32)

        qualia_candidate = (projector @ (gate * est.x_hat)).astype(np.float32)
        qualia = (
            float(health.trust) * qualia_candidate
            + (1.0 - float(health.trust)) * prev_q
        ).astype(np.float32)

        return QualiaState(
            gate=gate,
            qualia=qualia,
            precision=precision,
            observability=observability,
            body_coupling=body_coupling,
            value_grad=value_grad.astype(np.float32),
            habituation=habituation,
            trust_applied=float(health.trust),
            degraded=bool(health.degraded),
            reason=health.reason,
        )

    def _projector(self, state_dim: int) -> Matrix:
        if self.projector_matrix is None:
            return np.eye(state_dim, dtype=np.float32)
        projector = np.asarray(self.projector_matrix, dtype=np.float32)
        if projector.shape != (state_dim, state_dim):
            raise ValueError(f"projector_matrix shape must be {(state_dim, state_dim)}, got {projector.shape}")
        return projector

    def _precision(self, est: Estimate) -> Vector:
        diag = np.diag(est.cov).astype(np.float32)
        safe = np.where(diag > self.precision_floor, diag, self.precision_floor)
        return (1.0 / safe).astype(np.float32)

    def _observability(self, obs: Observation, est: Estimate) -> Vector:
        state_dim = est.x_hat.shape[0]
        observed_idx = np.flatnonzero(obs.mask.astype(bool))
        if observed_idx.size == 0:
            return np.zeros(state_dim, dtype=np.float32)
        H_obs = est.H[observed_idx, :].astype(np.float32)
        R_obs = obs.R[np.ix_(observed_idx, observed_idx)].astype(np.float32)
        solved = np.linalg.pinv(R_obs) @ H_obs
        info = H_obs.T @ solved
        return np.diag(info).astype(np.float32)

    def _body_coupling(
        self,
        layout: ObservationLayout,
        mask: NDArray[np.bool_],
        H: Matrix,
        state_dim: int,
    ) -> Vector:
        body_idx = layout.indices_of_kind("body", observed_only=True, mask=mask)
        if body_idx.size == 0:
            return np.zeros(state_dim, dtype=np.float32)
        body_rows = np.abs(H[body_idx, :]).astype(np.float32)
        return np.sum(body_rows, axis=0).astype(np.float32)

    def _habituation(self, prev_h: Vector, prev_q: Vector) -> Vector:
        return (
            self.habituation_decay * prev_h
            + (1.0 - self.habituation_decay) * np.abs(prev_q)
        ).astype(np.float32)


def _as_vector(values: Sequence[float], size: int, name: str) -> Vector:
    vector = np.asarray(list(values), dtype=np.float32).reshape(-1)
    if vector.size != size:
        raise ValueError(f"{name} length must match state_dim")
    return vector.astype(np.float32)


def _normalize(values: Vector) -> Vector:
    if values.size == 0:
        return values.astype(np.float32)
    clipped = np.log1p(np.maximum(values.astype(np.float32), 0.0))
    scale = float(np.max(clipped))
    if scale <= 1.0e-6:
        return np.zeros_like(clipped, dtype=np.float32)
    return (clipped / scale).astype(np.float32)


def _sigmoid(values: Vector) -> Vector:
    return (1.0 / (1.0 + np.exp(-values.astype(np.float32)))).astype(np.float32)
