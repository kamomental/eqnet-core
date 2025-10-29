from __future__ import annotations

import os
from copy import deepcopy
from types import SimpleNamespace

from emot_terrain_lab import observer


def _dummy_affect(valence: float, arousal: float) -> SimpleNamespace:
    return SimpleNamespace(valence=valence, arousal=arousal, confidence=0.88)


def _dummy_controls() -> SimpleNamespace:
    return SimpleNamespace(temperature=0.4, warmth=0.2)


def test_observer_markdown_disclaimer_varies_by_tone() -> None:
    base_state = {
        "offer_gate": {
            "ses": 0.32,
            "threshold": 0.55,
            "suggestion_allowed": False,
            "suppress_reason": "listen_bias",
            "recent_reject_s": 12.0,
            "community_bias": 0.05,
            "i_message_bias": 0.02,
        },
        "perceived_affect": {"valence": 0.0, "arousal": 0.0, "plutchik_top": []},
        "intent_hypotheses": [],
        "conflict_level": 0.1,
        "suggested_action": {"type": "listen_first_2s", "rationale": "listen_bias"},
        "evidence": {"cues": [], "eqnet_fields": {}},
    }

    prev_mode = os.environ.get("OBSERVER_DISCLAIMER_MODE")
    os.environ["OBSERVER_DISCLAIMER_MODE"] = "fixed_only"
    try:
        casual_text = observer.observer_markdown(deepcopy(base_state), tone="casual", culture="ja-JP")
        support_text = observer.observer_markdown(deepcopy(base_state), tone="support", culture="ja-JP")
        english_text = observer.observer_markdown(deepcopy(base_state), tone="neutral", culture="en-US")
    finally:
        if prev_mode is None:
            os.environ.pop("OBSERVER_DISCLAIMER_MODE", None)
        else:
            os.environ["OBSERVER_DISCLAIMER_MODE"] = prev_mode
    assert casual_text != support_text
    assert "hypothesis" in english_text


def test_infer_observer_state_safety_override_triggers() -> None:
    emergency_text = "助けて とても危険です。emergency please!"
    affect = _dummy_affect(0.05, 0.35)
    metrics = {"H": 0.42, "R": 0.38, "kappa": 0.21, "entropy": 0.57, "ignition": 0.6}
    controls = _dummy_controls()

    state = observer.infer_observer_state(
        user_text=emergency_text,
        affect=affect,
        metrics=metrics,
        controls=controls,
        gate_cfg={"threshold": 0.95},
        trust_score=0.1,
    )
    offer_gate = state["offer_gate"]
    assert offer_gate["suggestion_allowed"] is True
    assert offer_gate["suppress_reason"] == "safety_override"


def test_community_bias_lifts_ses() -> None:
    neutral_text = "最近ちょっと疲れていて、どうしたらいいかな？"
    affect = _dummy_affect(0.05, 0.1)
    metrics = {"H": 0.4, "R": 0.46, "kappa": 0.05, "entropy": 0.5, "ignition": 0.48}
    controls = _dummy_controls()

    no_bias_state = observer.infer_observer_state(
        user_text=neutral_text,
        affect=affect,
        metrics=metrics,
        controls=controls,
        gate_cfg={"community_bias": 0.0},
    )
    with_bias_state = observer.infer_observer_state(
        user_text=neutral_text,
        affect=affect,
        metrics=metrics,
        controls=controls,
        gate_cfg={"community_bias": 0.18},
    )
    assert with_bias_state["offer_gate"]["ses"] > no_bias_state["offer_gate"]["ses"]
    assert "community_bias" in with_bias_state["offer_gate"]
