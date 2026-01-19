from __future__ import annotations

from emot_terrain_lab.i18n.locale import lookup_text


def test_i18n_lookup_fallbacks_to_ja() -> None:
    expected = lookup_text("ja-JP", "presence_ack.short")
    fallback = lookup_text("zz-ZZ", "presence_ack.short")
    assert expected == fallback


def test_i18n_lookup_missing_key_returns_none() -> None:
    assert lookup_text("ja-JP", "presence_ack.missing") is None
