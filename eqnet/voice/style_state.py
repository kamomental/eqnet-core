"""Utterance style state definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, List


@dataclass(slots=True)
class UtteranceStyleState:
    """Represents the current speaking style for an utterance."""

    politeness: Literal["desu_masu", "da_dearu"]
    self_pronoun: str
    other_pronoun: str
    mood: Literal["calm", "happy", "excited", "tired", "serious"]
    sentence_length: Literal["short", "medium", "long"]
    filler_level: float
    colloquial_level: float
    tic_phrases: List[str]
    laughter_level: float
