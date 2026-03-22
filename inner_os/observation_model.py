from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float32]
Matrix = NDArray[np.float32]


@dataclass(frozen=True)
class ObservationChannelSpec:
    name: str
    kind: str
    index_slice: tuple[int, int]
    block_shape: tuple[int, ...]
    sample_rate_hz: float
    missing_policy: str = "masked"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObservationLayout:
    channels: tuple[ObservationChannelSpec, ...]

    @property
    def ext_slice(self) -> tuple[int, int]:
        return self.slice_for("ext")

    @property
    def body_slice(self) -> tuple[int, int]:
        return self.slice_for("body")

    @property
    def boundary_slice(self) -> tuple[int, int]:
        return self.slice_for("boundary")

    @property
    def action_slice(self) -> tuple[int, int]:
        return self.slice_for("action")

    def slice_for(self, name: str) -> tuple[int, int]:
        for channel in self.channels:
            if channel.name == name:
                return channel.index_slice
        return (0, 0)

    def channel_for(self, name: str) -> ObservationChannelSpec | None:
        for channel in self.channels:
            if channel.name == name:
                return channel
        return None

    def indices_of_kind(
        self,
        kind: str,
        *,
        observed_only: bool = False,
        mask: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.int_]:
        indices: list[int] = []
        for channel in self.channels:
            if channel.kind != kind:
                continue
            start, end = channel.index_slice
            indices.extend(range(start, end))
        result = np.asarray(indices, dtype=np.int_)
        if observed_only:
            if mask is None:
                raise ValueError("mask is required when observed_only=True")
            result = result[mask[result]]
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {"channels": [channel.to_dict() for channel in self.channels]}


# Backward-compatible alias for older imports.
ObservationChannelLayout = ObservationLayout


@dataclass(frozen=True)
class Observation:
    y: Vector
    mask: NDArray[np.bool_]
    R: Matrix
    dt: float
    timestamp: float
    layout: ObservationLayout

    def __post_init__(self) -> None:
        if float(self.dt) <= 0.0:
            raise ValueError("observation dt must be positive")
        if self.y.ndim != 1:
            raise ValueError("observation y must be a 1D vector")
        if self.mask.ndim != 1:
            raise ValueError("observation mask must be a 1D vector")
        if self.y.shape[0] != self.mask.shape[0]:
            raise ValueError("observation y and mask must have the same length")
        if self.R.shape != (self.y.shape[0], self.y.shape[0]):
            raise ValueError("observation R must match y length")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "y": self.y.tolist(),
            "mask": self.mask.astype(bool).tolist(),
            "R": self.R.tolist(),
            "dt": float(self.dt),
            "timestamp": float(self.timestamp),
            "layout": self.layout.to_dict(),
        }


class ObservationModel(Protocol):
    def encode(
        self,
        raw_sensors: Mapping[str, Any],
        body: Sequence[float],
        prev_action: Sequence[float],
        dt: float,
        *,
        timestamp: float = 0.0,
    ) -> Observation: ...

    def predict(
        self,
        x_hat: Sequence[float],
        body: Sequence[float],
        prev_action: Sequence[float],
    ) -> Vector: ...

    def jacobian(
        self,
        x_hat: Sequence[float],
        body: Sequence[float],
        prev_action: Sequence[float],
    ) -> Matrix: ...


@dataclass
class TensorObservationModel:
    latent_dim: int
    ext_size: int = 0
    body_size: int = 0
    boundary_size: int = 0
    action_size: int = 0
    ext_matrix: Matrix | None = None
    body_matrix: Matrix | None = None
    boundary_matrix: Matrix | None = None
    action_matrix: Matrix | None = None
    default_noise: float = 0.05
    channel_noise: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ext_matrix = self._coerce_matrix(self.ext_matrix, self.ext_size, 0)
        self.body_matrix = self._coerce_matrix(self.body_matrix, self.body_size, self.ext_size)
        self.boundary_matrix = self._coerce_matrix(
            self.boundary_matrix,
            self.boundary_size,
            self.ext_size + self.body_size,
        )
        self.action_matrix = self._coerce_matrix(
            self.action_matrix,
            self.action_size,
            self.ext_size + self.body_size + self.boundary_size,
        )
        total = self.ext_size + self.body_size + self.boundary_size + self.action_size
        if total <= 0:
            raise ValueError("observation dimension must be positive")
        self._total_observation_size = total

    def encode(
        self,
        raw_sensors: Mapping[str, Any],
        body: Sequence[float],
        prev_action: Sequence[float],
        dt: float,
        *,
        timestamp: float = 0.0,
    ) -> Observation:
        ext_values = _float_vector(raw_sensors.get("ext"), self.ext_size)
        body_values = _float_vector(raw_sensors.get("body_obs"), self.body_size)
        boundary_values = _float_vector(raw_sensors.get("boundary"), self.boundary_size)
        action_values = _float_vector(raw_sensors.get("action_feedback"), self.action_size)

        y = np.concatenate(
            [ext_values, body_values, boundary_values, action_values],
            dtype=np.float32,
        )
        mask = np.concatenate(
            [
                _mask_vector(raw_sensors.get("ext"), self.ext_size),
                _mask_vector(raw_sensors.get("body_obs"), self.body_size),
                _mask_vector(raw_sensors.get("boundary"), self.boundary_size),
                _mask_vector(raw_sensors.get("action_feedback"), self.action_size),
            ]
        ).astype(np.bool_)
        R = np.diag(
            np.concatenate(
                [
                    np.full(self.ext_size, self._noise_for("ext"), dtype=np.float32),
                    np.full(self.body_size, self._noise_for("body"), dtype=np.float32),
                    np.full(self.boundary_size, self._noise_for("boundary"), dtype=np.float32),
                    np.full(self.action_size, self._noise_for("action"), dtype=np.float32),
                ]
            )
        ).astype(np.float32)
        return Observation(
            y=y.astype(np.float32),
            mask=mask,
            R=R,
            dt=float(dt),
            timestamp=float(timestamp),
            layout=self._build_layout(float(dt)),
        )

    def predict(
        self,
        x_hat: Sequence[float],
        body: Sequence[float],
        prev_action: Sequence[float],
    ) -> Vector:
        x_vec = _as_vector(x_hat, self.latent_dim)
        body_vec = _float_vector(body, self.body_size)
        action_vec = _float_vector(prev_action, self.action_size)
        ext_pred = self.ext_matrix @ x_vec
        body_pred = self.body_matrix @ x_vec + body_vec
        boundary_pred = self.boundary_matrix @ x_vec
        action_pred = self.action_matrix @ x_vec + action_vec
        return np.concatenate([ext_pred, body_pred, boundary_pred, action_pred]).astype(np.float32)

    def jacobian(
        self,
        x_hat: Sequence[float],
        body: Sequence[float],
        prev_action: Sequence[float],
    ) -> Matrix:
        del x_hat, body, prev_action
        return np.vstack(
            [
                self.ext_matrix,
                self.body_matrix,
                self.boundary_matrix,
                self.action_matrix,
            ]
        ).astype(np.float32)

    def _coerce_matrix(self, matrix: Matrix | None, rows: int, offset: int) -> Matrix:
        if rows <= 0:
            return np.zeros((0, self.latent_dim), dtype=np.float32)
        if matrix is not None:
            coerced = np.asarray(matrix, dtype=np.float32)
            if coerced.shape != (rows, self.latent_dim):
                raise ValueError(f"matrix shape must be {(rows, self.latent_dim)}, got {coerced.shape}")
            return coerced
        base = np.zeros((rows, self.latent_dim), dtype=np.float32)
        for index in range(rows):
            base[index, (offset + index) % self.latent_dim] = 1.0
        return base

    def _noise_for(self, channel: str) -> float:
        try:
            return float(self.channel_noise.get(channel, self.default_noise))
        except (TypeError, ValueError):
            return float(self.default_noise)

    def _build_layout(self, dt: float) -> ObservationLayout:
        sample_rate_hz = 1.0 / dt if dt > 0.0 else 0.0
        ext_slice = (0, self.ext_size)
        body_slice = (self.ext_size, self.ext_size + self.body_size)
        boundary_slice = (
            self.ext_size + self.body_size,
            self.ext_size + self.body_size + self.boundary_size,
        )
        action_slice = (
            self.ext_size + self.body_size + self.boundary_size,
            self._total_observation_size,
        )
        return ObservationLayout(
            channels=(
                ObservationChannelSpec(
                    name="ext",
                    kind="external",
                    index_slice=ext_slice,
                    block_shape=(self.ext_size,),
                    sample_rate_hz=sample_rate_hz,
                ),
                ObservationChannelSpec(
                    name="body",
                    kind="body",
                    index_slice=body_slice,
                    block_shape=(self.body_size,),
                    sample_rate_hz=sample_rate_hz,
                ),
                ObservationChannelSpec(
                    name="boundary",
                    kind="boundary",
                    index_slice=boundary_slice,
                    block_shape=(self.boundary_size,),
                    sample_rate_hz=sample_rate_hz,
                ),
                ObservationChannelSpec(
                    name="action",
                    kind="action",
                    index_slice=action_slice,
                    block_shape=(self.action_size,),
                    sample_rate_hz=sample_rate_hz,
                ),
            )
        )


def _as_vector(values: Sequence[float], size: int) -> Vector:
    vector = np.asarray(list(values), dtype=np.float32).reshape(-1)
    if vector.size < size:
        pad = np.zeros(size - vector.size, dtype=np.float32)
        vector = np.concatenate([vector, pad])
    elif vector.size > size:
        vector = vector[:size]
    return vector.astype(np.float32)


def _float_vector(values: Any, size: int) -> Vector:
    if size <= 0:
        return np.zeros(0, dtype=np.float32)
    if values is None:
        return np.zeros(size, dtype=np.float32)
    if isinstance(values, (int, float)):
        return _as_vector([float(values)], size)
    if isinstance(values, np.ndarray):
        return _as_vector(values.tolist(), size)
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return _as_vector([0.0 if item is None else float(item) for item in values], size)
    return np.zeros(size, dtype=np.float32)


def _mask_vector(values: Any, size: int) -> NDArray[np.bool_]:
    if size <= 0:
        return np.zeros(0, dtype=np.bool_)
    if values is None:
        return np.zeros(size, dtype=np.bool_)
    if isinstance(values, (int, float)):
        mask = np.zeros(size, dtype=np.bool_)
        mask[0] = True
        return mask
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        mask = np.zeros(size, dtype=np.bool_)
        limit = min(len(values), size)
        for index in range(limit):
            mask[index] = values[index] is not None
        return mask
    return np.zeros(size, dtype=np.bool_)
