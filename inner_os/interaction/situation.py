from __future__ import annotations

from .models import RelationalMood, SituationState


def summarize_situation_state(
    *,
    affective_summary: dict[str, float],
    current_risks: list[str],
    active_goals: list[str],
    selection_reasons: list[str],
    relation_bias_strength: float = 0.0,
) -> SituationState:
    social_tension = float(affective_summary.get("social_tension", 0.0))
    trust = float(affective_summary.get("trust", 0.0))
    shared_attention = min(1.0, relation_bias_strength * 0.72 + max(0.0, trust) * 0.18)
    repair_window_open = "repair" in active_goals or "social" in selection_reasons
    if current_risks:
        current_phase = "guarded"
    elif repair_window_open:
        current_phase = "repair"
    elif relation_bias_strength >= 0.28:
        current_phase = "check_in"
    else:
        current_phase = "ongoing"
    return SituationState(
        scene_mode="co_present" if relation_bias_strength > 0.0 else "ambient",
        repair_window_open=repair_window_open,
        shared_attention=round(shared_attention, 4),
        social_pressure=round(social_tension, 4),
        continuity_weight=round(relation_bias_strength, 4),
        current_phase=current_phase,
    )


def derive_relational_mood(
    *,
    affective_summary: dict[str, float],
    situation_state: SituationState,
    partner_address_hint: str = "",
    partner_timing_hint: str = "",
    partner_stance_hint: str = "",
) -> RelationalMood:
    trust = max(0.0, float(affective_summary.get("trust", 0.0)))
    curiosity = max(0.0, float(affective_summary.get("curiosity", 0.0)))
    arousal = max(0.0, float(affective_summary.get("arousal", 0.0)))
    social_tension = max(0.0, float(affective_summary.get("social_tension", 0.0)))

    future_pull = min(1.0, curiosity * 0.34 + trust * 0.24 + situation_state.continuity_weight * 0.46)
    care = min(1.0, trust * 0.42 + max(0.0, 1.0 - social_tension) * 0.16 + situation_state.continuity_weight * 0.34)
    shared_world_pull = min(1.0, situation_state.shared_attention * 0.72 + trust * 0.12 + curiosity * 0.08)
    confidence_signal = min(1.0, trust * 0.28 + arousal * 0.12 + max(0.0, 1.0 - social_tension) * 0.24)

    reverence = 0.0
    innocence = 0.0
    if partner_stance_hint == "respectful":
        reverence = min(1.0, 0.42 + trust * 0.16 + situation_state.continuity_weight * 0.18)
    if partner_timing_hint == "delayed":
        reverence = min(1.0, reverence + 0.14)
    if partner_address_hint == "companion" and partner_stance_hint == "familiar":
        innocence = min(1.0, 0.24 + curiosity * 0.18 + max(0.0, 1.0 - confidence_signal) * 0.18)
        future_pull = min(1.0, future_pull + 0.12)
        shared_world_pull = min(1.0, shared_world_pull + 0.12)

    if social_tension > 0.58:
        future_pull *= 0.82
        confidence_signal *= 0.76

    return RelationalMood(
        future_pull=round(future_pull, 4),
        reverence=round(reverence, 4),
        innocence=round(innocence, 4),
        care=round(care, 4),
        shared_world_pull=round(shared_world_pull, 4),
        confidence_signal=round(confidence_signal, 4),
    )
