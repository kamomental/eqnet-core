from __future__ import annotations

from typing import Any, Mapping

from .models import InteractionTrace, LiveInteractionRegulation, RelationalMood, SituationState


def derive_live_interaction_regulation(
    *,
    current_state: Mapping[str, Any],
    situation_state: SituationState,
    relational_mood: RelationalMood,
    interaction_trace: InteractionTrace | None = None,
) -> LiveInteractionRegulation:
    replay_intensity = _float(current_state.get("replay_intensity"), 0.0)
    interaction_afterglow = _float(current_state.get("interaction_afterglow"), 0.0)
    continuity_score = _float(current_state.get("continuity_score"), 0.0)
    anticipation_tension = _float(current_state.get("anticipation_tension"), 0.0)
    meaning_inertia = _float(current_state.get("meaning_inertia"), 0.0)
    pending_meaning = _float(current_state.get("pending_meaning"), 0.0)
    relational_clarity = _float(current_state.get("relational_clarity"), 0.0)
    long_term_theme_strength = _float(current_state.get("long_term_theme_strength"), 0.0)
    caution_bias = _float(current_state.get("caution_bias"), 0.0)

    past_loop_pull = _clamp01(
        replay_intensity * 0.56
        + interaction_afterglow * 0.26
        + continuity_score * 0.18
    )
    future_loop_pull = _clamp01(
        anticipation_tension * 0.42
        + relational_mood.future_pull * 0.38
        + relational_mood.shared_world_pull * 0.2
    )
    fantasy_loop_pull = _clamp01(
        meaning_inertia * 0.42
        + pending_meaning * 0.28
        + (1.0 - relational_clarity) * 0.2
        + long_term_theme_strength * 0.1
    )
    shared_attention_active = _clamp01(
        situation_state.shared_attention * 0.68
        + relational_mood.shared_world_pull * 0.22
        + (_float(getattr(interaction_trace, "shared_attention", 0.0), 0.0) * 0.1 if interaction_trace else 0.0)
    )
    strained_pause = _clamp01(
        situation_state.social_pressure * 0.4
        + caution_bias * 0.24
        + (_float(getattr(interaction_trace, "repair_signal", 0.0), 0.0) * 0.08 if interaction_trace else 0.0)
        + (0.18 if interaction_trace and interaction_trace.pause_mode in {"measured_ritual", "waiting"} else 0.0)
    )

    if relational_mood.care >= 0.58:
        distance_expectation = "holding_space"
    elif relational_mood.future_pull >= 0.52:
        distance_expectation = "future_opening"
    elif relational_mood.reverence >= 0.42:
        distance_expectation = "respectful_distance"
    else:
        distance_expectation = "gentle_near"

    cues: list[str] = []
    if past_loop_pull >= 0.38:
        cues.append("past_loop_active")
    if future_loop_pull >= 0.38:
        cues.append("future_loop_active")
    if fantasy_loop_pull >= 0.34:
        cues.append("fantasy_loop_active")
    if shared_attention_active >= 0.45:
        cues.append("shared_attention_active")
    if strained_pause >= 0.42:
        cues.append("strained_pause")
    if situation_state.repair_window_open:
        cues.append("repair_window_open")
    cues.append(f"distance_expectation_{distance_expectation}")

    return LiveInteractionRegulation(
        past_loop_pull=round(past_loop_pull, 4),
        future_loop_pull=round(future_loop_pull, 4),
        fantasy_loop_pull=round(fantasy_loop_pull, 4),
        shared_attention_active=round(shared_attention_active, 4),
        strained_pause=round(strained_pause, 4),
        repair_window_open=situation_state.repair_window_open,
        distance_expectation=distance_expectation,
        cues=cues,
    )


def derive_memory_context_live_regulation(
    *,
    memory_context: Any | None,
    situation_state: SituationState,
    relational_mood: RelationalMood,
) -> LiveInteractionRegulation:
    relation_bias = _float(getattr(memory_context, "relation_bias_strength", 0.0), 0.0) if memory_context is not None else 0.0
    continuity_threads = len(getattr(memory_context, "continuity_threads", [])) if memory_context is not None else 0
    retention_summary = len(getattr(memory_context, "retention_summary", [])) if memory_context is not None else 0
    semantic_hints = getattr(memory_context, "semantic_hints", []) if memory_context is not None else []
    semantic_weight = 0.0
    for hint in semantic_hints[:3]:
        semantic_weight += _float(getattr(hint, "recurrence_weight", 0.0), 0.0)
    semantic_weight = min(1.0, semantic_weight)
    current_state = {
        "replay_intensity": min(1.0, continuity_threads * 0.16 + retention_summary * 0.08),
        "interaction_afterglow": relation_bias * 0.24,
        "continuity_score": min(1.0, continuity_threads * 0.18 + relation_bias * 0.28),
        "anticipation_tension": relation_bias * 0.22 + relational_mood.future_pull * 0.18,
        "meaning_inertia": semantic_weight * 0.3,
        "pending_meaning": semantic_weight * 0.24,
        "relational_clarity": relation_bias * 0.42,
        "long_term_theme_strength": semantic_weight * 0.18,
        "caution_bias": relational_mood.reverence * 0.24,
    }
    return derive_live_interaction_regulation(
        current_state=current_state,
        situation_state=situation_state,
        relational_mood=relational_mood,
    )


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
