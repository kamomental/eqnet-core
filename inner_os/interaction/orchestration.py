from __future__ import annotations

from .live_regulation import derive_live_interaction_regulation
from .models import (
    LiveInteractionRegulation,
    NonverbalProfile,
    RelationalMood,
    SituationState,
)


def orchestrate_interaction(
    *,
    current_risks: list[str],
    situation_state: SituationState,
    relational_mood: RelationalMood,
    nonverbal_profile: NonverbalProfile,
    live_regulation: LiveInteractionRegulation | None = None,
) -> dict[str, object]:
    regulation = live_regulation or derive_live_interaction_regulation(
        current_state={},
        situation_state=situation_state,
        relational_mood=relational_mood,
    )
    orchestration_mode = "attune"
    dominant_driver = "shared_attention"
    if "danger" in current_risks:
        orchestration_mode = "contain"
        dominant_driver = "stability"
    elif regulation.repair_window_open and regulation.strained_pause >= 0.38:
        orchestration_mode = "repair"
        dominant_driver = "care"
    elif relational_mood.future_pull >= 0.48 and regulation.future_loop_pull >= 0.34:
        orchestration_mode = "advance"
        dominant_driver = "future"
    elif regulation.fantasy_loop_pull >= 0.38 or regulation.past_loop_pull >= 0.4:
        orchestration_mode = "reflect"
        dominant_driver = "memory"

    coherence_score = _clamp01(
        situation_state.shared_attention * 0.28
        + relational_mood.shared_world_pull * 0.18
        + relational_mood.care * 0.16
        + relational_mood.confidence_signal * 0.12
        + (1.0 - regulation.strained_pause) * 0.14
        + (0.12 if nonverbal_profile.gaze_mode in {"shared_attention_hold", "soft_hold", "gentle_return"} else 0.0)
    )
    contact_readiness = _clamp01(
        relational_mood.future_pull * 0.24
        + relational_mood.care * 0.18
        + regulation.shared_attention_active * 0.22
        + coherence_score * 0.18
        - regulation.strained_pause * 0.14
    )
    human_presence_signal = _clamp01(
        coherence_score * 0.34
        + regulation.shared_attention_active * 0.22
        + regulation.past_loop_pull * 0.12
        + regulation.future_loop_pull * 0.12
        + regulation.fantasy_loop_pull * 0.08
        + relational_mood.care * 0.12
    )

    cues: list[str] = [
        f"orchestration_{orchestration_mode}",
        f"driver_{dominant_driver}",
    ]
    if contact_readiness >= 0.48:
        cues.append("contact_ready")
    if human_presence_signal >= 0.5:
        cues.append("presence_dense")

    return {
        "orchestration_mode": orchestration_mode,
        "dominant_driver": dominant_driver,
        "coherence_score": round(coherence_score, 4),
        "contact_readiness": round(contact_readiness, 4),
        "human_presence_signal": round(human_presence_signal, 4),
        "distance_strategy": regulation.distance_expectation,
        "repair_bias": regulation.repair_window_open,
        "cues": cues,
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
