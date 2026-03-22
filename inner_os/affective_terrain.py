from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Sequence

import numpy as np

from .affective_position import AffectivePositionState
from .observation_model import Matrix, Vector


@dataclass(frozen=True)
class AffectiveTerrainState:
    centers: Matrix
    widths: Vector
    value_weights: Vector
    approach_weights: Vector
    avoid_weights: Vector
    protect_weights: Vector
    anchor_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.centers.ndim != 2:
            raise ValueError("centers must be a 2D matrix")
        patch_count, _ = self.centers.shape
        expected = (patch_count,)
        for name, values in (
            ("widths", self.widths),
            ("value_weights", self.value_weights),
            ("approach_weights", self.approach_weights),
            ("avoid_weights", self.avoid_weights),
            ("protect_weights", self.protect_weights),
        ):
            if values.shape != expected:
                raise ValueError(f"{name} shape must match patch count {expected}")
        if self.anchor_labels and len(self.anchor_labels) != patch_count:
            raise ValueError("anchor_labels length must match patch count")

    @property
    def patch_count(self) -> int:
        return int(self.centers.shape[0])

    @property
    def position_dim(self) -> int:
        return int(self.centers.shape[1])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "centers": self.centers.tolist(),
            "widths": self.widths.tolist(),
            "value_weights": self.value_weights.tolist(),
            "approach_weights": self.approach_weights.tolist(),
            "avoid_weights": self.avoid_weights.tolist(),
            "protect_weights": self.protect_weights.tolist(),
            "anchor_labels": list(self.anchor_labels),
        }


@dataclass(frozen=True)
class TerrainReadout:
    value: float
    grad: Vector
    curvature: Vector
    approach_bias: float
    avoid_bias: float
    protect_bias: float
    active_patch_index: int
    active_patch_label: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": float(self.value),
            "grad": self.grad.tolist(),
            "curvature": self.curvature.tolist(),
            "approach_bias": float(self.approach_bias),
            "avoid_bias": float(self.avoid_bias),
            "protect_bias": float(self.protect_bias),
            "active_patch_index": int(self.active_patch_index),
            "active_patch_label": self.active_patch_label,
        }


class AffectiveTerrain(Protocol):
    def read(
        self,
        terrain_state: AffectiveTerrainState,
        position_state: AffectivePositionState,
    ) -> TerrainReadout: ...


@dataclass
class BasicAffectiveTerrain:
    width_floor: float = 1.0e-3

    def read(
        self,
        terrain_state: AffectiveTerrainState,
        position_state: AffectivePositionState,
    ) -> TerrainReadout:
        if terrain_state.position_dim != position_state.position_dim:
            raise ValueError("terrain position dimension must match affective position dimension")

        activations, deltas, widths = self._activations(terrain_state, position_state.z_aff)
        value = float(np.sum(terrain_state.value_weights * activations))

        width_sq = np.square(widths).astype(np.float32)
        grad_terms = (
            terrain_state.value_weights[:, None]
            * activations[:, None]
            * (-deltas / width_sq[:, None])
        )
        grad = np.sum(grad_terms, axis=0).astype(np.float32)

        curvature_terms = (
            terrain_state.value_weights[:, None]
            * activations[:, None]
            * ((np.square(deltas) / np.square(width_sq)[:, None]) - (1.0 / width_sq[:, None]))
        )
        curvature = np.sum(curvature_terms, axis=0).astype(np.float32)

        norm = float(np.sum(activations))
        if norm <= 1.0e-6:
            approach_bias = 0.0
            avoid_bias = 0.0
            protect_bias = 0.0
        else:
            weights = activations / norm
            approach_bias = float(np.sum(weights * terrain_state.approach_weights))
            avoid_bias = float(np.sum(weights * terrain_state.avoid_weights))
            protect_bias = float(np.sum(weights * terrain_state.protect_weights))

        active_patch_index = int(np.argmax(activations))
        active_patch_label = (
            terrain_state.anchor_labels[active_patch_index]
            if terrain_state.anchor_labels
            else f"patch_{active_patch_index}"
        )

        return TerrainReadout(
            value=value,
            grad=grad,
            curvature=curvature,
            approach_bias=_clamp01(approach_bias),
            avoid_bias=_clamp01(avoid_bias),
            protect_bias=_clamp01(protect_bias),
            active_patch_index=active_patch_index,
            active_patch_label=active_patch_label,
        )

    def _activations(
        self,
        terrain_state: AffectiveTerrainState,
        z_aff: Vector,
    ) -> tuple[Vector, Matrix, Vector]:
        widths = np.maximum(terrain_state.widths.astype(np.float32), self.width_floor)
        deltas = (z_aff.astype(np.float32)[None, :] - terrain_state.centers.astype(np.float32)).astype(np.float32)
        squared = np.sum(np.square(deltas), axis=1).astype(np.float32)
        denom = 2.0 * np.square(widths)
        activations = np.exp(-squared / denom).astype(np.float32)
        return activations, deltas, widths


def make_neutral_affective_terrain_state(
    *,
    position_dim: int,
    patch_count: int = 4,
) -> AffectiveTerrainState:
    if position_dim <= 0:
        raise ValueError("position_dim must be positive")
    if patch_count <= 0:
        raise ValueError("patch_count must be positive")
    centers = np.zeros((patch_count, position_dim), dtype=np.float32)
    widths = np.full(patch_count, 1.0, dtype=np.float32)
    zeros = np.zeros(patch_count, dtype=np.float32)
    labels = tuple(f"patch_{index}" for index in range(patch_count))
    return AffectiveTerrainState(
        centers=centers,
        widths=widths,
        value_weights=zeros.copy(),
        approach_weights=zeros.copy(),
        avoid_weights=zeros.copy(),
        protect_weights=zeros.copy(),
        anchor_labels=labels,
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
