from __future__ import annotations

import numpy as np

from emot_terrain_lab.memory.reference_helper import (
    ReplayCandidate,
    ReplayOutcome,
    ReferenceResolution,
    compose_recall_response,
    handle_memory_reference,
    resolve_reference,
    search_memory,
)
from emot_terrain_lab.terrain.memory_palace import MemoryPalace, MemoryNode


class FakeSystem:
    def __init__(self) -> None:
        node = MemoryNode("garden", "default", [0.5, 0.5])
        self.memory_palace = MemoryPalace([node])
        self.memory_palace.labels["garden"][-1] = (
            '2019 Kyoto trip where we walked quietly and I said "let us rest today."'
        )
        trace = np.ones(node.capacity, dtype=float) * 0.6
        self.memory_palace.traces["garden"] = trace
        qualia = self.memory_palace.qualia_state["garden"]
        qualia["energy"] = 0.7
        qualia["magnitude"] = 0.6


def test_resolve_reference_extracts_year_and_keywords() -> None:
    ref = resolve_reference("Do you remember the 2019 Kyoto trip when you let me rest?")
    assert ref.when_year == 2019
    assert ref.what == "travel"
    assert ref.who == "you"


def test_search_memory_returns_high_score_for_matching_label() -> None:
    system = FakeSystem()
    ref = resolve_reference("Do you remember the walk during our 2019 Kyoto trip?")
    candidates = search_memory(system, ref, k=1)
    assert candidates
    cand = candidates[0]
    assert cand.score > 0.4
    assert cand.anchor >= 0.5


def test_handle_memory_reference_high_fidelity_reply() -> None:
    system = FakeSystem()
    result = handle_memory_reference(
        system,
        "During the 2019 Kyoto trip when I was really tired, what did I say again?",
        tone="support",
        culture="ja-JP",
        max_reply_chars=140,
    )
    assert result["fidelity"] >= 0.45
    assert "Kyoto" in result["reply"] or "2019" in result["reply"]
    assert "いっしょ" in result["reply"]
    assert result["candidate"]["node"] == "garden"
    assert result.get("meta", {}).get("mode") == "recall"


def _build_outcome(fidelity: float, anchor: float, score: float = 0.5) -> ReplayOutcome:
    candidate = ReplayCandidate(
        node_name="garden",
        label="quiet walk and you said to rest",
        semantic=0.6,
        affective=0.55,
        anchor=anchor,
        score=score,
    )
    return ReplayOutcome(candidate=candidate, fidelity=fidelity, affect_strength=0.5)


def test_compose_recall_response_interpret_mid_band() -> None:
    outcome = _build_outcome(fidelity=0.55, anchor=0.3)
    ref = ReferenceResolution(who=None, when_year=None, where=None, what=None, summary="remember Kyoto 2019?")
    reply, fidelity, label, source = compose_recall_response(
        outcome,
        ref,
        tone="support",
        culture="ja-JP",
        strategy="interpret",
        max_reply_chars=160,
    )
    assert fidelity == 0.55
    assert label == "quiet walk and you said to rest"
    assert "教えて" in reply
    assert source in {"llm", "template", "fixed", "unknown"}


def test_compose_recall_response_mend_includes_feedback() -> None:
    outcome = _build_outcome(fidelity=0.72, anchor=0.6, score=0.7)
    ref = ReferenceResolution(who=None, when_year=2019, where="Kyoto", what="travel", summary="Kyoto trip 2019")
    reply, fidelity, _, _ = compose_recall_response(
        outcome,
        ref,
        tone="support",
        culture="en-US",
        strategy="mend",
        user_feedback="It was actually in spring 2018",
    )
    assert "spring 2018" in reply
    assert "I'm" in reply or "I’m" in reply
    assert fidelity == 0.72


def test_compose_recall_response_respects_max_chars() -> None:
    outcome = _build_outcome(fidelity=0.68, anchor=0.55, score=0.65)
    ref = ReferenceResolution(who=None, when_year=None, where=None, what=None, summary="memory")
    reply, _, _, _ = compose_recall_response(
        outcome,
        ref,
        tone="support",
        culture="ja-JP",
        strategy="recall",
        max_reply_chars=40,
    )
    assert len(reply) <= 40
