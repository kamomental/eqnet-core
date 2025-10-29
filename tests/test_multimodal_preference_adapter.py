from __future__ import annotations

from emot_terrain_lab.audio.asr_pipeline import SpeechSegment
from emot_terrain_lab.persona.multimodal_adapter import MultimodalPreferenceBuilder


def make_segment(text: str, speaker: str = "speaker_0", confidence: float = 1.0) -> SpeechSegment:
    return SpeechSegment(start=0.0, end=1.0, text=text, speaker=speaker, confidence=confidence)


def test_builder_combines_modalities() -> None:
    builder = MultimodalPreferenceBuilder()
    draft = builder.build(
        text_chunks=["丁寧にお願いします。"],
        speech_segments=[make_segment("断定は避けてください。", speaker="user")],
        image_captions=["静かな夜の写真。ゆっくり話したい気分。"],
        lang_hint="ja-JP",
    )
    assert draft.profile["persona"]["culture"] == "ja-jp"
    assert draft.profile["persona"]["tone"] in {"support", "polite"}
    assert any("multimodal_sections" in note for note in draft.notes)


def test_builder_requires_content() -> None:
    builder = MultimodalPreferenceBuilder()
    try:
        builder.build()
    except ValueError:
        pass
    else:  # pragma: no cover - should not happen
        raise AssertionError("Expected ValueError when no content is supplied.")
