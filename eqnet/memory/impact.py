"""Impact scoring helpers for nightly promotion."""

from __future__ import annotations

from typing import Protocol

EMO_IMPACT = {
    "joy": 0.6,
    "trust": 0.7,
    "surprise": 0.75,
    "anger": 0.85,
    "sadness": 0.85,
    "fear": 0.9,
}

AROUSAL_DIM = 11


class TerrainState(Protocol):
    def similarity_to_existing_themes(self, topic: str | None) -> float: ...

    def sensitivity_for_topic(self, topic: str | None) -> float: ...


def compute_impact(moment: object, terrain_state: TerrainState | None) -> float:
    """Return a coarse 0-1 impact score for ``moment``.

    ``moment`` is expected to provide ``emotion_tag`` and ``qualia_vec``.
    ``terrain_state`` can be ``None`` during early bootstrapping.
    """

    emo_tag = getattr(moment, "emotion_tag", None)
    emo_weight = EMO_IMPACT.get(emo_tag, 0.5)

    qualia_vec = getattr(moment, "qualia_vec", None)
    arousal = 0.0
    if qualia_vec is not None and len(qualia_vec) > AROUSAL_DIM:
        try:
            arousal = float(qualia_vec[AROUSAL_DIM])
        except (TypeError, ValueError):
            arousal = 0.0
    arousal = max(0.0, min(1.0, arousal))

    topic = getattr(moment, "topic", None)
    if terrain_state is None:
        novelty = 0.5
        terrain = 0.5
    else:
        novelty = 1.0 - float(terrain_state.similarity_to_existing_themes(topic))
        terrain = float(terrain_state.sensitivity_for_topic(topic))

    base = emo_weight * (0.5 + 0.5 * arousal)
    impact = base * (0.5 + 0.5 * novelty) * (0.5 + 0.5 * terrain)
    return max(0.0, min(1.0, impact))
