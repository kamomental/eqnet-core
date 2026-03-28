from __future__ import annotations

from emot_terrain_lab.i18n.locale import lookup_text, lookup_value


def test_i18n_lookup_fallbacks_to_ja() -> None:
    expected = lookup_text("ja-JP", "presence_ack.short")
    fallback = lookup_text("zz-ZZ", "presence_ack.short")
    assert expected == fallback


def test_i18n_lookup_missing_key_returns_none() -> None:
    assert lookup_text("ja-JP", "presence_ack.missing") is None


def test_i18n_lookup_can_return_non_string_values() -> None:
    value = lookup_value("ja-JP", "inner_os.content_policy_cues.opening_request.contains_any")
    assert isinstance(value, list)
    assert "切り出" in value
