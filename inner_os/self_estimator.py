from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from .observation_model import Matrix, Observation, ObservationModel, Vector


@dataclass(frozen=True)
class Estimate:
    x_hat: Vector
    cov: Matrix
    y_hat: Vector
    innovation: Vector
    innovation_cov: Matrix
    H: Matrix
    nis: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x_hat": self.x_hat.tolist(),
            "cov": self.cov.tolist(),
            "y_hat": self.y_hat.tolist(),
            "innovation": self.innovation.tolist(),
            "innovation_cov": self.innovation_cov.tolist(),
            "H": self.H.tolist(),
            "nis": float(self.nis),
        }


@dataclass(frozen=True)
class EstimatorHealth:
    innovation_norm: float
    nis: float
    observability_mean: float
    observed_fraction: float
    trust: float
    degraded: bool
    reason: str
    overconfident_estimate: bool
    observation_contract_break: bool
    low_observability: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SelfEstimator(Protocol):
    def step(
        self,
        obs: Observation,
        body: Sequence[float],
        prev_action: Sequence[float],
    ) -> Estimate: ...


@dataclass
class ResidualLinearSelfEstimator:
    observation_model: ObservationModel
    latent_dim: int
    A_x: Matrix | None = None
    Q: Matrix | None = None
    initial_state: Vector | None = None
    initial_cov: Matrix | None = None

    def __post_init__(self) -> None:
        self.A_x = self._coerce_square(self.A_x, fill_identity=True)
        self.Q = self._coerce_square(self.Q, fill_identity=False, scale=0.02)
        self._x_hat = (
            np.asarray(self.initial_state, dtype=np.float32).reshape(-1)
            if self.initial_state is not None
            else np.zeros(self.latent_dim, dtype=np.float32)
        )
        if self._x_hat.size != self.latent_dim:
            raise ValueError("initial_state size must match latent_dim")
        self._cov = self._coerce_square(self.initial_cov, fill_identity=False, scale=1.0)

    def step(
        self,
        obs: Observation,
        body: Sequence[float],
        prev_action: Sequence[float],
    ) -> Estimate:
        x_pred = (self.A_x @ self._x_hat).astype(np.float32)
        cov_pred = (self.A_x @ self._cov @ self.A_x.T + self.Q).astype(np.float32)

        y_pred = self.observation_model.predict(x_pred, body, prev_action).astype(np.float32)
        H = self.observation_model.jacobian(x_pred, body, prev_action).astype(np.float32)
        if y_pred.ndim != 1:
            raise ValueError("predicted observation must be a 1D vector")
        if y_pred.shape != obs.y.shape:
            raise ValueError("predicted observation shape must match observation y")
        if H.shape != (obs.y.shape[0], self.latent_dim):
            raise ValueError("jacobian shape must match observation rows and latent_dim")
        if obs.R.shape != (obs.y.shape[0], obs.y.shape[0]):
            raise ValueError("observation covariance shape must match observation y")
        innovation = (obs.y - y_pred).astype(np.float32)

        observed_idx = np.flatnonzero(obs.mask.astype(bool))
        if observed_idx.size == 0:
            self._x_hat = x_pred
            self._cov = cov_pred
            innovation_cov = np.zeros((0, 0), dtype=np.float32)
            estimate = Estimate(
                x_hat=self._x_hat.copy(),
                cov=self._cov.copy(),
                y_hat=y_pred,
                innovation=innovation,
                innovation_cov=innovation_cov,
                H=H,
                nis=0.0,
            )
            return estimate

        H_obs = H[observed_idx, :]
        innovation_obs = innovation[observed_idx]
        R_obs = obs.R[np.ix_(observed_idx, observed_idx)].astype(np.float32)

        innovation_cov = (H_obs @ cov_pred @ H_obs.T + R_obs).astype(np.float32)
        innovation_cov_inv = np.linalg.pinv(innovation_cov).astype(np.float32)
        kalman_gain = (cov_pred @ H_obs.T @ innovation_cov_inv).astype(np.float32)

        x_next = (x_pred + kalman_gain @ innovation_obs).astype(np.float32)
        identity = np.eye(self.latent_dim, dtype=np.float32)
        joseph = identity - kalman_gain @ H_obs
        cov_next = (
            joseph @ cov_pred @ joseph.T + kalman_gain @ R_obs @ kalman_gain.T
        ).astype(np.float32)
        cov_next = _symmetrize(cov_next)

        nis = float(innovation_obs.T @ innovation_cov_inv @ innovation_obs)

        self._x_hat = x_next
        self._cov = cov_next
        return Estimate(
            x_hat=x_next.copy(),
            cov=cov_next.copy(),
            y_hat=y_pred,
            innovation=innovation,
            innovation_cov=innovation_cov,
            H=H,
            nis=nis,
        )

    def _coerce_square(
        self,
        matrix: Matrix | None,
        *,
        fill_identity: bool,
        scale: float = 1.0,
    ) -> Matrix:
        if matrix is None:
            base = np.eye(self.latent_dim, dtype=np.float32) if fill_identity else np.eye(self.latent_dim, dtype=np.float32) * scale
            return base.astype(np.float32)
        coerced = np.asarray(matrix, dtype=np.float32)
        if coerced.shape != (self.latent_dim, self.latent_dim):
            raise ValueError(f"matrix shape must be {(self.latent_dim, self.latent_dim)}, got {coerced.shape}")
        return coerced


def evaluate_estimator_health(
    estimate: Estimate,
    obs: Observation | None = None,
    *,
    obs_mismatch_threshold: float = 1.0,
    nis_threshold: float = 12.0,
    min_observability_mean: float = 0.01,
) -> EstimatorHealth:
    innovation_norm = float(np.linalg.norm(estimate.innovation))
    if estimate.innovation_cov.size:
        diag = np.diag(estimate.innovation_cov).astype(np.float32)
        safe_diag = np.where(diag > 1.0e-6, diag, 1.0e-6)
        observability_mean = float(np.mean(1.0 / safe_diag))
    else:
        observability_mean = 0.0
    observed_fraction = 0.0
    if obs is not None and obs.mask.size:
        observed_fraction = float(np.mean(obs.mask.astype(np.float32)))

    overconfident_estimate = bool(estimate.nis > nis_threshold and np.mean(np.diag(estimate.cov)) < 0.25)
    observation_contract_break = bool(innovation_norm > obs_mismatch_threshold)
    low_observability = bool(observability_mean < min_observability_mean)
    predict_only_mode = bool(obs is not None and not np.any(obs.mask))
    degraded = bool(overconfident_estimate or observation_contract_break or low_observability or predict_only_mode)
    reasons: list[str] = []
    if predict_only_mode:
        reasons.append("predict_only")
    if overconfident_estimate:
        reasons.append("overconfident_estimate")
    if observation_contract_break:
        reasons.append("observation_contract_break")
    if low_observability:
        reasons.append("low_observability")
    reason = ",".join(reasons) if reasons else "healthy"
    trust = _clamp01(
        1.0
        - 0.35 * min(1.0, innovation_norm / max(obs_mismatch_threshold, 1.0e-6))
        - 0.35 * min(1.0, float(estimate.nis) / max(nis_threshold, 1.0e-6))
        - 0.15 * (1.0 - observed_fraction)
        - (0.1 if low_observability else 0.0)
    )

    return EstimatorHealth(
        innovation_norm=innovation_norm,
        nis=float(estimate.nis),
        observability_mean=observability_mean,
        observed_fraction=observed_fraction,
        trust=trust,
        degraded=degraded,
        reason=reason,
        overconfident_estimate=overconfident_estimate,
        observation_contract_break=observation_contract_break,
        low_observability=low_observability,
    )


def _symmetrize(matrix: Matrix) -> Matrix:
    return ((matrix + matrix.T) * 0.5).astype(np.float32)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
