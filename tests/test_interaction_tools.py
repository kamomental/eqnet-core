from __future__ import annotations

from eqnet.runtime.interaction_tools import (
    build_reflex_signal,
    estimate_resonance_state,
    interaction_digest,
    shape_response_profile,
)


def test_estimate_resonance_state_returns_belief_fields() -> None:
    state = estimate_resonance_state(
        text="不安で怖いです...",
        prev_state={"valence": 0.5, "arousal": 0.5, "safety": 0.5},
        policy=None,
    )
    assert isinstance(state.get("valence"), float)
    assert isinstance(state.get("arousal"), float)
    assert isinstance(state.get("safety"), float)
    assert isinstance(state.get("confidence"), float)
    assert isinstance(state.get("reason_codes"), list)
    assert float(state.get("safety") or 0.0) < 0.5


def test_reflex_and_response_shaper_contract() -> None:
    resonance = {
        "valence": 0.3,
        "arousal": 0.8,
        "safety": 0.2,
        "confidence": 0.7,
    }
    reflex = build_reflex_signal(resonance=resonance, policy=None)
    shaper = shape_response_profile(
        resonance=resonance,
        metabolism={"attention_budget_used": 0.35, "resource_budget": {"attention": {"level": 0.2}}},
        policy=None,
    )
    assert isinstance(reflex.get("mode"), str)
    assert isinstance(reflex.get("text"), str)
    assert isinstance(reflex.get("latency_target_ms"), int)
    assert isinstance(shaper.get("mode"), str)
    assert isinstance(shaper.get("pace"), str)
    assert isinstance(shaper.get("strategy"), str)
    assert isinstance(shaper.get("max_sentences"), int)
    assert int(shaper.get("max_sentences") or 0) >= 1


def test_interaction_digest_stable_for_same_payload() -> None:
    payload = {"resonance": {"valence": 0.5}, "reflex": {"mode": "neutral"}}
    assert interaction_digest(payload) == interaction_digest(payload)
