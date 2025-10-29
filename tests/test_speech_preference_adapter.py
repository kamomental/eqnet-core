from __future__ import annotations

import pytest

from emot_terrain_lab.audio.asr_pipeline import SpeechSegment
from emot_terrain_lab.persona.speech_adapter import SpeechPreferenceExtractor


def make_segment(text: str, speaker: str = "speaker_0", confidence: float = 0.9) -> SpeechSegment:
    return SpeechSegment(start=0.0, end=1.0, text=text, speaker=speaker, confidence=confidence)


def test_aggregate_text_filters_by_speaker_and_confidence() -> None:
    segments = [
        make_segment("まず丁寧に聴いてほしい", speaker="A"),
        make_segment("助言は短く一つで", speaker="B"),
        make_segment("confidence low", speaker="B", confidence=0.1),
    ]
    extractor = SpeechPreferenceExtractor(target_speakers={"B"}, min_confidence=0.5)
    text = extractor.aggregate_text(segments)
    assert text == "助言は短く一つで"


def test_to_persona_returns_draft() -> None:
    segments = [
        make_segment("丁寧に聴いてください。断定はしないでください。", speaker="user"),
        make_segment("これは背景説明。", speaker="observer"),
    ]
    extractor = SpeechPreferenceExtractor(target_speakers={"user"})
    draft = extractor.to_persona(segments, lang_hint="ja-JP")
    assert draft.profile["persona"]["culture"] == "ja-jp"
    assert draft.profile["persona"]["tone"] in {"support", "polite"}
    assert "断定" in draft.profile["preferences"]["taboo"]


def test_to_persona_raises_when_no_text() -> None:
    extractor = SpeechPreferenceExtractor()
    with pytest.raises(ValueError):
        extractor.to_persona([], lang_hint="en-US")
