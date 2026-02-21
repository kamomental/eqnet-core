from __future__ import annotations

from eqnet.memory.state_vector import (
    TemporalStateVector,
    coherence_score,
    status_lexicon,
    temporal_delta,
)


def test_temporal_state_vector_roundtrip():
    src = TemporalStateVector(
        timestamp_ms=1000,
        valence=0.2,
        arousal=-0.1,
        value_tags={"safety": 0.7, "growth": 0.4},
        open_loops=0.3,
        event_scale=0.5,
        embed=(0.1, 0.2),
    )
    payload = src.to_dict()
    dst = TemporalStateVector.from_dict(payload)
    assert dst.timestamp_ms == 1000
    assert dst.value_tags["safety"] == 0.7
    assert list(dst.embed) == [0.1, 0.2]


def test_coherence_score_drops_on_large_unexpected_change():
    prev = TemporalStateVector(timestamp_ms=1000, valence=0.1, arousal=0.1, event_scale=0.2)
    curr = TemporalStateVector(timestamp_ms=2000, valence=1.0, arousal=1.0, event_scale=0.2)
    assert temporal_delta(prev, curr) > 0.3
    assert coherence_score(prev, curr) < 0.5


def test_to_status_text_ja_includes_focus_and_stability():
    tags = {"tag_a": 0.8, "tag_b": 0.6}
    state = TemporalStateVector(
        timestamp_ms=1,
        value_tags=tags,
        open_loops=0.4,
    )
    text = state.to_status_text(coherence=0.75, sat_ratio=0.2, locale="ja")
    lex = status_lexicon("ja")
    for tag in tags:
        assert tag in text
    assert lex["unresolved_mid"] in text
    assert lex["stability_stable"] in text


def test_to_status_text_handles_empty_tags():
    state = TemporalStateVector(timestamp_ms=1, value_tags={}, open_loops=0.0)
    text = state.to_status_text(coherence=None, sat_ratio=None, locale="ja")
    lex = status_lexicon("ja")
    assert lex["focus_balanced"] in text
