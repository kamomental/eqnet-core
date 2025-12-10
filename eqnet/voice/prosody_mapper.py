from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from .context import ProsodyVec


class EmotionProsodyMapper(ABC):
    """Voice Field §9 Emotion-to-Prosody Bridge interface."""

    @abstractmethod
    def map(self, q_t: np.ndarray, voice_id: str) -> ProsodyVec:
        ...
