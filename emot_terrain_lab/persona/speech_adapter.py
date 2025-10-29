# -*- coding: utf-8 -*-
"""
Utilities to convert recognised speech segments into persona preference drafts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Set

from emot_terrain_lab.audio.asr_pipeline import SpeechSegment
from emot_terrain_lab.persona.profile_input import PersonaDraft, persona_from_text


@dataclass
class SpeechPreferenceExtractor:
    """
    Aggregate speech segments for a subset of speakers and derive a persona
    profile via ``persona_from_text``.
    """

    target_speakers: Optional[Set[str]] = None
    min_confidence: float = 0.0
    join_with: str = " "
    notes: Sequence[str] = field(default_factory=list)

    def aggregate_text(self, segments: Iterable[SpeechSegment]) -> str:
        chunks = []
        for seg in segments:
            if self.target_speakers and seg.speaker not in self.target_speakers:
                continue
            if seg.confidence < self.min_confidence:
                continue
            cleaned = seg.text.strip()
            if cleaned:
                chunks.append(cleaned)
        return self.join_with.join(chunks).strip()

    def to_persona(
        self,
        segments: Iterable[SpeechSegment],
        *,
        lang_hint: Optional[str] = None,
    ) -> PersonaDraft:
        text = self.aggregate_text(segments)
        if not text:
            raise ValueError("No speech text available after filtering.")
        draft = persona_from_text(text, lang_hint=lang_hint)
        if self.notes:
            draft.notes.extend(self.notes)
        draft.notes.append(f"speech_segments_used={len(text.split())}")
        return draft


__all__ = ["SpeechPreferenceExtractor"]
