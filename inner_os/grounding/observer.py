from __future__ import annotations

from typing import Mapping, Any

from .models import ObservationBundle, GroundedEntity


def observe(input_streams: Mapping[str, Any]) -> ObservationBundle:
    """観測入力を最小の ObservationBundle へ正規化する。"""
    labels = list(input_streams.get("entity_labels") or [])
    entity_attributes = list(input_streams.get("entity_attributes") or [])
    entities = [
        GroundedEntity(
            entity_id=f"obs:{idx}",
            label=str(label),
            source="mock",
            confidence=0.6,
            uncertainty=0.4,
            attributes=dict(entity_attributes[idx]) if idx < len(entity_attributes) and isinstance(entity_attributes[idx], Mapping) else {},
        )
        for idx, label in enumerate(labels)
    ]
    uncertainty = 0.8 if not entities else 0.35
    return ObservationBundle(
        entities=entities,
        observation_uncertainty=uncertainty,
        notes=["mock_observation_bundle"],
    )
