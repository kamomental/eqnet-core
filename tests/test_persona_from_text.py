from __future__ import annotations

from emot_terrain_lab.persona.profile_input import persona_from_text


def test_persona_from_text_japanese_polite():
    text = "丁寧に聴いてほしいです。助言は一度に1つで、断定や説教調は避けてください。"
    draft = persona_from_text(text)
    profile = draft.profile

    assert profile["persona"]["culture"] == "ja-jp"
    assert profile["persona"]["tone"] in {"support", "polite"}
    assert "断定" in profile["preferences"]["taboo"]
    assert profile["preferences"]["advice_style"] == "one_point"
    assert profile["constraints"]["always_hypothesis"] is True


def test_persona_from_text_english_casual():
    text = "Please keep it casual, friendly, and concise. One actionable tip is enough. No lecturing tone."
    draft = persona_from_text(text, lang_hint="en-US")
    profile = draft.profile

    assert profile["persona"]["culture"] == "en-us"
    assert profile["persona"]["tone"] in {"casual", "support"}
    assert "lecturing" in profile["preferences"]["taboo"]
    assert profile["preferences"]["advice_style"] == "one_point"
