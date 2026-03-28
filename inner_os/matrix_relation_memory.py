from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


FeatureMap = Mapping[str, float]


def _as_float32_matrix(value: Any, *, label: str) -> NDArray[np.float32]:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{label} must be a rank-2 matrix")
    return matrix.astype(np.float32, copy=False)


def _as_float32_vector(value: Any, *, label: str) -> NDArray[np.float32]:
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1:
        raise ValueError(f"{label} must be a rank-1 vector")
    return vector.astype(np.float32, copy=False)


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _sigmoid(value: float) -> float:
    clipped = float(np.clip(value, -40.0, 40.0))
    return float(1.0 / (1.0 + np.exp(-clipped)))


def _apply_nonlinearity(
    value: NDArray[np.float32],
    *,
    mode: str,
    max_abs_state: float,
) -> NDArray[np.float32]:
    if mode == "tanh":
        activated = np.tanh(value)
    elif mode == "relu":
        activated = np.maximum(value, 0.0)
    elif mode == "identity":
        activated = value.astype(np.float32, copy=False)
    else:
        raise ValueError(f"unsupported matrix relation memory nonlinearity: {mode}")
    if max_abs_state > 0.0:
        activated = np.clip(activated, -max_abs_state, max_abs_state)
    return activated.astype(np.float32, copy=False)


@dataclass(frozen=True)
class MatrixMemoryHeadSpec:
    """入力特徴から行列記憶ヘッドを更新する投影定義。"""

    name: str
    key_projection: NDArray[np.float32]
    value_projection: NDArray[np.float32]
    query_projection: NDArray[np.float32]
    forget_projection: NDArray[np.float32]
    forget_bias: float = 0.0
    update_scale: float = 1.0
    read_scale: float = 1.0

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        if not name:
            raise ValueError("matrix memory head name must not be empty")
        key_projection = _as_float32_matrix(self.key_projection, label="key_projection")
        value_projection = _as_float32_matrix(self.value_projection, label="value_projection")
        query_projection = _as_float32_matrix(self.query_projection, label="query_projection")
        forget_projection = _as_float32_vector(self.forget_projection, label="forget_projection")
        if key_projection.shape[1] != value_projection.shape[1]:
            raise ValueError("key_projection and value_projection must share feature width")
        if query_projection.shape[1] != key_projection.shape[1]:
            raise ValueError("query_projection and key_projection must share feature width")
        if query_projection.shape[0] != key_projection.shape[0]:
            raise ValueError("query_projection output width must match key_projection output width")
        if forget_projection.shape[0] != key_projection.shape[1]:
            raise ValueError("forget_projection width must match feature width")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "key_projection", key_projection)
        object.__setattr__(self, "value_projection", value_projection)
        object.__setattr__(self, "query_projection", query_projection)
        object.__setattr__(self, "forget_projection", forget_projection)
        object.__setattr__(self, "forget_bias", float(self.forget_bias))
        object.__setattr__(self, "update_scale", float(self.update_scale))
        object.__setattr__(self, "read_scale", float(self.read_scale))

    @property
    def feature_dim(self) -> int:
        return int(self.key_projection.shape[1])

    @property
    def key_dim(self) -> int:
        return int(self.key_projection.shape[0])

    @property
    def value_dim(self) -> int:
        return int(self.value_projection.shape[0])


@dataclass(frozen=True)
class MatrixRelationMemoryConfig:
    """M²RNN 的な行列記憶の最小構成。"""

    feature_names: tuple[str, ...]
    head_specs: tuple[MatrixMemoryHeadSpec, ...]
    nonlinearity: str = "tanh"
    max_abs_state: float = 4.0
    config_version: str = "v1"

    def __post_init__(self) -> None:
        feature_names = tuple(str(name).strip() for name in self.feature_names if str(name).strip())
        if not feature_names:
            raise ValueError("matrix relation memory feature_names must not be empty")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("matrix relation memory feature_names must be unique")
        head_specs = tuple(self.head_specs)
        if not head_specs:
            raise ValueError("matrix relation memory requires at least one head")
        head_names = [spec.name for spec in head_specs]
        if len(set(head_names)) != len(head_names):
            raise ValueError("matrix relation memory head names must be unique")
        feature_dim = len(feature_names)
        key_dim = head_specs[0].key_dim
        value_dim = head_specs[0].value_dim
        for spec in head_specs:
            if spec.feature_dim != feature_dim:
                raise ValueError("all matrix memory heads must match feature_names width")
            if spec.key_dim != key_dim:
                raise ValueError("all matrix memory heads must share key/query width")
            if spec.value_dim != value_dim:
                raise ValueError("all matrix memory heads must share value width")
        if self.nonlinearity not in {"tanh", "relu", "identity"}:
            raise ValueError("matrix relation memory nonlinearity must be tanh, relu, or identity")
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "head_specs", head_specs)
        object.__setattr__(self, "nonlinearity", str(self.nonlinearity))
        object.__setattr__(self, "max_abs_state", max(0.0, float(self.max_abs_state)))
        object.__setattr__(self, "config_version", str(self.config_version or "v1"))

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    @property
    def key_dim(self) -> int:
        return self.head_specs[0].key_dim

    @property
    def value_dim(self) -> int:
        return self.head_specs[0].value_dim


@dataclass(frozen=True)
class MatrixMemoryHeadState:
    """1ヘッド分の online 行列記憶状態。"""

    name: str
    matrix: NDArray[np.float32]
    retain_gate: float = 0.0
    update_norm: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name or "").strip())
        object.__setattr__(self, "matrix", _as_float32_matrix(self.matrix, label="matrix"))
        object.__setattr__(self, "retain_gate", _clamp01(float(self.retain_gate)))
        object.__setattr__(self, "update_norm", max(0.0, float(self.update_norm)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "retain_gate": float(self.retain_gate),
            "update_norm": float(self.update_norm),
            "shape": [int(self.matrix.shape[0]), int(self.matrix.shape[1])],
            "state_norm": float(np.linalg.norm(self.matrix)),
        }


@dataclass(frozen=True)
class MatrixRelationMemoryState:
    """全ヘッドの行列記憶と直近特徴の保持。"""

    config_version: str
    feature_names: tuple[str, ...]
    head_states: tuple[MatrixMemoryHeadState, ...]
    feature_snapshot: dict[str, float] = field(default_factory=dict)
    step_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_version", str(self.config_version or "v1"))
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "head_states", tuple(self.head_states))
        object.__setattr__(
            self,
            "feature_snapshot",
            {
                str(name): float(value)
                for name, value in dict(self.feature_snapshot).items()
                if str(name).strip()
            },
        )
        object.__setattr__(self, "step_count", max(0, int(self.step_count)))

    def head_state(self, name: str) -> MatrixMemoryHeadState:
        target = str(name or "").strip()
        for head_state in self.head_states:
            if head_state.name == target:
                return head_state
        raise KeyError(f"matrix memory head not found: {target}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_version": self.config_version,
            "feature_names": list(self.feature_names),
            "feature_snapshot": {
                str(name): float(value)
                for name, value in self.feature_snapshot.items()
            },
            "step_count": int(self.step_count),
            "head_states": [head_state.to_dict() for head_state in self.head_states],
        }


@dataclass(frozen=True)
class MatrixMemoryReadout:
    """query に対する各ヘッドの読出結果。"""

    head_scores: dict[str, float]
    head_vectors: dict[str, tuple[float, ...]]
    combined_vector: tuple[float, ...]
    dominant_head: str
    winner_margin: float
    query_snapshot: dict[str, float] = field(default_factory=dict)

    @property
    def combined_score(self) -> float:
        return float(np.linalg.norm(np.asarray(self.combined_vector, dtype=np.float32)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "head_scores": {
                str(name): float(score)
                for name, score in self.head_scores.items()
            },
            "head_vectors": {
                str(name): list(vector)
                for name, vector in self.head_vectors.items()
            },
            "combined_vector": list(self.combined_vector),
            "combined_score": float(self.combined_score),
            "dominant_head": self.dominant_head,
            "winner_margin": float(self.winner_margin),
            "query_snapshot": {
                str(name): float(value)
                for name, value in self.query_snapshot.items()
            },
        }


class MatrixRelationMemoryCore:
    """outer product による関係行列記憶の最小核。"""

    def __init__(self, config: MatrixRelationMemoryConfig) -> None:
        self._config = config

    @property
    def config(self) -> MatrixRelationMemoryConfig:
        return self._config

    def initialize(self) -> MatrixRelationMemoryState:
        head_states = tuple(
            MatrixMemoryHeadState(
                name=spec.name,
                matrix=np.zeros((spec.key_dim, spec.value_dim), dtype=np.float32),
            )
            for spec in self._config.head_specs
        )
        return MatrixRelationMemoryState(
            config_version=self._config.config_version,
            feature_names=self._config.feature_names,
            head_states=head_states,
        )

    def update(
        self,
        state: MatrixRelationMemoryState,
        *,
        features: FeatureMap,
    ) -> MatrixRelationMemoryState:
        self._validate_state(state)
        feature_vector, snapshot = self._encode_features(features)
        head_states: list[MatrixMemoryHeadState] = []
        for spec, previous_state in zip(self._config.head_specs, state.head_states):
            retain_gate = _sigmoid(float(spec.forget_projection @ feature_vector) + spec.forget_bias)
            key_vector = spec.key_projection @ feature_vector
            value_vector = spec.value_projection @ feature_vector
            delta = np.outer(key_vector, value_vector).astype(np.float32) * spec.update_scale
            raw_state = retain_gate * previous_state.matrix + delta
            updated_matrix = _apply_nonlinearity(
                raw_state,
                mode=self._config.nonlinearity,
                max_abs_state=self._config.max_abs_state,
            )
            head_states.append(
                MatrixMemoryHeadState(
                    name=spec.name,
                    matrix=updated_matrix,
                    retain_gate=retain_gate,
                    update_norm=float(np.linalg.norm(delta)),
                )
            )
        return MatrixRelationMemoryState(
            config_version=self._config.config_version,
            feature_names=self._config.feature_names,
            head_states=tuple(head_states),
            feature_snapshot=snapshot,
            step_count=state.step_count + 1,
        )

    def read(
        self,
        state: MatrixRelationMemoryState,
        *,
        query_features: FeatureMap,
    ) -> MatrixMemoryReadout:
        self._validate_state(state)
        feature_vector, snapshot = self._encode_features(query_features)
        head_scores: dict[str, float] = {}
        head_vectors: dict[str, tuple[float, ...]] = {}
        combined = np.zeros((self._config.value_dim,), dtype=np.float32)
        for spec, head_state in zip(self._config.head_specs, state.head_states):
            query_vector = spec.query_projection @ feature_vector
            read_vector = (query_vector @ head_state.matrix).astype(np.float32) * spec.read_scale
            combined += read_vector
            head_vectors[spec.name] = tuple(float(item) for item in read_vector.tolist())
            head_scores[spec.name] = float(np.linalg.norm(read_vector))
        ordered_heads = sorted(head_scores.items(), key=lambda item: item[1], reverse=True)
        dominant_head = ordered_heads[0][0] if ordered_heads else ""
        top_score = ordered_heads[0][1] if ordered_heads else 0.0
        second_score = ordered_heads[1][1] if len(ordered_heads) > 1 else 0.0
        winner_margin = max(0.0, float(top_score - second_score))
        return MatrixMemoryReadout(
            head_scores=head_scores,
            head_vectors=head_vectors,
            combined_vector=tuple(float(item) for item in combined.tolist()),
            dominant_head=dominant_head,
            winner_margin=winner_margin,
            query_snapshot=snapshot,
        )

    def _validate_state(self, state: MatrixRelationMemoryState) -> None:
        if state.config_version != self._config.config_version:
            raise ValueError("matrix relation memory state/config version mismatch")
        if tuple(state.feature_names) != self._config.feature_names:
            raise ValueError("matrix relation memory feature_names mismatch")
        if len(state.head_states) != len(self._config.head_specs):
            raise ValueError("matrix relation memory head count mismatch")
        expected_names = tuple(spec.name for spec in self._config.head_specs)
        actual_names = tuple(head_state.name for head_state in state.head_states)
        if actual_names != expected_names:
            raise ValueError("matrix relation memory head ordering mismatch")

    def _encode_features(self, features: FeatureMap) -> tuple[NDArray[np.float32], dict[str, float]]:
        snapshot: dict[str, float] = {}
        values = np.zeros((self._config.feature_dim,), dtype=np.float32)
        payload = dict(features or {})
        for index, name in enumerate(self._config.feature_names):
            try:
                numeric = float(payload.get(name, 0.0))
            except (TypeError, ValueError):
                numeric = 0.0
            values[index] = numeric
            snapshot[name] = float(numeric)
        return values, snapshot
