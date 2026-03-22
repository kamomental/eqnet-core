from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np

from .observation_model import Matrix, Vector


@dataclass(frozen=True)
class AffectivePositionState:
    z_aff: Vector
    cov: Matrix
    confidence: float
    source_weights: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.z_aff.ndim != 1:
            raise ValueError("z_aff must be a 1D vector")
        if self.cov.shape != (self.z_aff.shape[0], self.z_aff.shape[0]):
            raise ValueError("cov shape must match z_aff length")

    @property
    def position_dim(self) -> int:
        return int(self.z_aff.shape[0])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "z_aff": self.z_aff.tolist(),
            "cov": self.cov.tolist(),
            "confidence": float(self.confidence),
            "source_weights": {
                str(key): float(max(0.0, min(1.0, value)))
                for key, value in self.source_weights.items()
            },
        }


def make_neutral_affective_position(position_dim: int) -> AffectivePositionState:
    if position_dim <= 0:
        raise ValueError("position_dim must be positive")
    z_aff = np.zeros(position_dim, dtype=np.float32)
    cov = np.eye(position_dim, dtype=np.float32)
    return AffectivePositionState(
        z_aff=z_aff,
        cov=cov,
        confidence=0.0,
        source_weights={"carryover": 1.0},
    )
