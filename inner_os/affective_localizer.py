from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .affective_position import AffectivePositionState, make_neutral_affective_position
from .observation_model import Matrix, Vector
from .qualia_projector import QualiaState
from .self_estimator import Estimate, EstimatorHealth


class AffectiveLocalizer(Protocol):
    def localize(
        self,
        *,
        estimate: Estimate,
        health: EstimatorHealth,
        qualia_state: QualiaState,
        memory: Sequence[float],
        prev_position: AffectivePositionState | None,
        dt: float,
    ) -> AffectivePositionState: ...


@dataclass
class BasicAffectiveLocalizer:
    position_dim: int
    state_gain: float = 0.42
    qualia_gain: float = 0.38
    memory_gain: float = 0.2
    carryover: float = 0.72
    state_projection: Matrix | None = None
    qualia_projection: Matrix | None = None

    def localize(
        self,
        *,
        estimate: Estimate,
        health: EstimatorHealth,
        qualia_state: QualiaState,
        memory: Sequence[float],
        prev_position: AffectivePositionState | None,
        dt: float,
    ) -> AffectivePositionState:
        if dt <= 0.0:
            raise ValueError("affective localizer dt must be positive")

        prev = prev_position or make_neutral_affective_position(self.position_dim)
        state_component = self._project_state(estimate.x_hat, estimate.x_hat.shape[0], self.state_projection)
        qualia_component = self._project_state(
            qualia_state.qualia,
            qualia_state.qualia.shape[0],
            self.qualia_projection,
        ) * float(qualia_state.trust_applied)
        memory_component = self._reduce_memory(memory)

        raw_position = (
            self.state_gain * state_component
            + self.qualia_gain * qualia_component
            + self.memory_gain * memory_component
        ).astype(np.float32)

        carry = self._carryover(health, qualia_state)
        z_aff = (carry * prev.z_aff + (1.0 - carry) * raw_position).astype(np.float32)

        projected_cov = self._project_covariance(estimate.cov)
        uncertainty = np.eye(self.position_dim, dtype=np.float32) * (1.0 - float(health.trust))
        cov = (carry * prev.cov + (1.0 - carry) * projected_cov + 0.1 * uncertainty).astype(np.float32)
        cov = _symmetrize(cov)

        source_weights = _normalize_weights(
            {
                "state": float(np.linalg.norm(self.state_gain * state_component)),
                "qualia": float(np.linalg.norm(self.qualia_gain * qualia_component)),
                "memory": float(np.linalg.norm(self.memory_gain * memory_component)),
                "carryover": float(np.linalg.norm(carry * prev.z_aff)),
            }
        )
        confidence = self._confidence(health, qualia_state, cov)
        return AffectivePositionState(
            z_aff=z_aff,
            cov=cov,
            confidence=confidence,
            source_weights=source_weights,
        )

    def _project_state(
        self,
        values: Sequence[float],
        input_dim: int,
        projection: Matrix | None,
    ) -> Vector:
        vector = np.asarray(list(values), dtype=np.float32).reshape(-1)
        if vector.size != input_dim:
            raise ValueError("input vector length must match input_dim")
        matrix = projection
        if matrix is None:
            matrix = _default_projection(self.position_dim, input_dim)
        if matrix.shape != (self.position_dim, input_dim):
            raise ValueError("projection shape must match position_dim and input_dim")
        return (matrix @ vector).astype(np.float32)

    def _reduce_memory(self, memory: Sequence[float]) -> Vector:
        vector = np.asarray(list(memory), dtype=np.float32).reshape(-1)
        if vector.size == 0:
            return np.zeros(self.position_dim, dtype=np.float32)
        if vector.size == self.position_dim:
            return vector.astype(np.float32)
        splits = np.array_split(vector, self.position_dim)
        reduced = np.asarray(
            [float(np.mean(chunk)) if chunk.size else 0.0 for chunk in splits],
            dtype=np.float32,
        )
        return reduced

    def _project_covariance(self, cov: Matrix) -> Matrix:
        state_dim = cov.shape[0]
        projection = self.state_projection
        if projection is None:
            projection = _default_projection(self.position_dim, state_dim)
        if projection.shape != (self.position_dim, state_dim):
            raise ValueError("state projection shape must match position and state dimensions")
        projected = (projection @ cov.astype(np.float32) @ projection.T).astype(np.float32)
        return _symmetrize(projected)

    def _carryover(self, health: EstimatorHealth, qualia_state: QualiaState) -> float:
        trust = 0.5 * float(health.trust) + 0.5 * float(qualia_state.trust_applied)
        carry = self.carryover + 0.18 * (1.0 - trust)
        if qualia_state.degraded or health.degraded:
            carry += 0.08
        return _clamp01(carry)

    def _confidence(self, health: EstimatorHealth, qualia_state: QualiaState, cov: Matrix) -> float:
        trace = float(np.trace(cov)) / max(1, cov.shape[0])
        certainty = 1.0 / (1.0 + trace)
        trust = 0.55 * float(health.trust) + 0.45 * float(qualia_state.trust_applied)
        if qualia_state.degraded or health.degraded:
            trust *= 0.8
        return _clamp01(trust * certainty)


def _default_projection(output_dim: int, input_dim: int) -> Matrix:
    projection = np.zeros((output_dim, input_dim), dtype=np.float32)
    for row in range(output_dim):
        projection[row, row % input_dim] = 1.0
    return projection


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(max(0.0, value) for value in weights.values()))
    if total <= 1.0e-6:
        return {key: 0.0 for key in weights}
    return {key: float(max(0.0, value) / total) for key, value in weights.items()}


def _symmetrize(matrix: Matrix) -> Matrix:
    return ((matrix + matrix.T) * 0.5).astype(np.float32)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
