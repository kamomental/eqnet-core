# -*- coding: utf-8 -*-
"""Core data structure for Five-Sense-First representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Sequence

Source = Literal["external", "simulated", "recalled"]
Modality = Literal[
    "vision",
    "audio",
    "haptic",
    "thermal",
    "olfactory",
    "gustatory",
]


@dataclass
class SenseEnvelope:
    """Simple, serialisable container for sense features."""

    id: str
    modality: Modality
    features: Dict[str, float] = field(default_factory=dict)
    latent: Sequence[float] | None = None
    confidence: float = 0.5
    source: Source = "external"
    t_tau: float = 0.0
    tags: List[str] = field(default_factory=list)

    def feature(self, name: str, default: float = 0.0) -> float:
        """Return a named feature with a default."""
        return float(self.features.get(name, default))

    def to_dict(self) -> Dict[str, object]:
        """Serialise for receipts."""
        return {
            "id": self.id,
            "modality": self.modality,
            "features": {k: float(v) for k, v in self.features.items()},
            "confidence": float(self.confidence),
            "source": self.source,
            "t_tau": float(self.t_tau),
            "tags": list(self.tags),
        }


def clamp_features(features: Dict[str, float]) -> Dict[str, float]:
    """Clamp raw feature dictionary into [0, 1] range."""
    clamped = {}
    for key, value in features.items():
        try:
            clamped[key] = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            continue
    return clamped


__all__ = ["SenseEnvelope", "Source", "Modality", "clamp_features"]
