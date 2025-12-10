from __future__ import annotations
import numpy as np
from ..prosody_mapper import EmotionProsodyMapper
from ..context import ProsodyVec


class SimpleRuleBasedMapper(EmotionProsodyMapper):
    """Voice Field §9 Emotion-to-Prosody Bridge v0 (provisional)."""

    def map(self, q_t: np.ndarray, voice_id: str) -> ProsodyVec:
        valence = float(q_t[0])
        arousal = float(q_t[1])
        stress = float(q_t[2])

        tempo = 1.0 + 0.2 * arousal
        pitch = 0.5 * valence
        energy = 0.5 + 0.4 * (arousal - stress)
        energy = max(0.2, min(1.0, energy))
        return ProsodyVec(tempo=tempo, pitch=pitch, energy=energy)
