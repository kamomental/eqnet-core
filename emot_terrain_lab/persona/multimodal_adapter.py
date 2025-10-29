# -*- coding: utf-8 -*-
"""
Combine multiple modalities (text, speech, images) into a persona profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from emot_terrain_lab.audio.asr_pipeline import SpeechSegment
from emot_terrain_lab.persona.profile_input import PersonaDraft, persona_from_text
from emot_terrain_lab.persona.speech_adapter import SpeechPreferenceExtractor


@dataclass
class MultimodalPreferenceBuilder:
    speech_extractor: SpeechPreferenceExtractor = field(default_factory=SpeechPreferenceExtractor)
    join_with: str = "\n"
    extra_notes: Sequence[str] = field(default_factory=list)

    def build(
        self,
        *,
        text_chunks: Optional[Iterable[str]] = None,
        speech_segments: Optional[Iterable[SpeechSegment]] = None,
        image_captions: Optional[Iterable[str]] = None,
        lang_hint: Optional[str] = None,
    ) -> PersonaDraft:
        parts: list[str] = []

        if text_chunks:
            for chunk in text_chunks:
                cleaned = (chunk or "").strip()
                if cleaned:
                    parts.append(cleaned)

        if speech_segments:
            speech_text = self.speech_extractor.aggregate_text(speech_segments)
            if speech_text:
                parts.append(speech_text)

        if image_captions:
            for caption in image_captions:
                cleaned = (caption or "").strip()
                if cleaned:
                    parts.append(cleaned)

        if not parts:
            raise ValueError("No textual content supplied for persona derivation.")

        combined = self.join_with.join(parts).strip()
        draft = persona_from_text(combined, lang_hint=lang_hint)
        draft.notes.append(f"multimodal_sections={len(parts)}")
        draft.notes.extend(self.extra_notes)
        return draft


__all__ = ["MultimodalPreferenceBuilder"]
