from __future__ import annotations

from .models import NonverbalProfile, RelationalMood, SituationState


def compose_nonverbal_profile(
    *,
    utterance_stance: str,
    affective_summary: dict[str, float],
    situation_state: SituationState,
    relational_mood: RelationalMood,
    partner_address_hint: str = "",
    partner_timing_hint: str = "",
    partner_stance_hint: str = "",
) -> NonverbalProfile:
    social_tension = float(affective_summary.get("social_tension", 0.0))
    arousal = float(affective_summary.get("arousal", 0.0))
    cues: list[str] = []

    gaze_mode = "steady"
    pause_mode = "neutral"
    proximity_mode = "neutral"
    silence_mode = "neutral"
    gesture_mode = "contained"

    if utterance_stance == "warm_check_in":
        gaze_mode = "soft_hold"
        pause_mode = "short_warm"
        proximity_mode = "welcoming"
        silence_mode = "shared"
        gesture_mode = "open_small"
    elif utterance_stance == "measured_check_in":
        gaze_mode = "respectful_glance"
        pause_mode = "measured"
        proximity_mode = "respectful_distance"
        silence_mode = "careful"
        gesture_mode = "minimal"
    elif utterance_stance == "gentle_check_in":
        gaze_mode = "gentle_return"
        pause_mode = "soft"
        proximity_mode = "soft_approach"
        silence_mode = "patient"
        gesture_mode = "contained_open"

    if partner_timing_hint == "delayed" or situation_state.social_pressure > 0.58:
        pause_mode = "waiting"
        silence_mode = "respectful_wait"
        cues.append("leave_space_before_reply")
    if partner_address_hint == "companion" and (
        situation_state.shared_attention >= 0.45 or partner_stance_hint == "familiar"
    ):
        gaze_mode = "shared_attention_hold"
        cues.append("shared_attention_bias")
    if partner_stance_hint == "respectful":
        proximity_mode = "respectful_distance"
        gesture_mode = "minimal"
    if social_tension >= 0.55:
        pause_mode = "longer_regulating"
        silence_mode = "tension_buffer"
        cues.append("tension_buffer_pause")
    if arousal <= 0.25 and utterance_stance != "neutral_observation":
        cues.append("soft_entry_breath")
    if situation_state.repair_window_open:
        cues.append("repair_window_open")
        silence_mode = "repair_sensitive"
    if relational_mood.future_pull >= 0.55:
        proximity_mode = "future_opening"
        cues.append("future_pull_opening")
    if relational_mood.reverence >= 0.4:
        pause_mode = "measured_ritual"
        gaze_mode = "respectful_glance"
        proximity_mode = "respectful_distance"
        cues.append("reverent_spacing")
    if relational_mood.innocence >= 0.28:
        silence_mode = "tender_tentative"
        gesture_mode = "small_sincere"
        cues.append("innocent_hesitation")
    if relational_mood.care >= 0.58:
        pause_mode = "patient_care"
        silence_mode = "holding_space"
        cues.append("care_holding")
    if relational_mood.shared_world_pull >= 0.24:
        gaze_mode = "shared_attention_hold"
        cues.append("shared_world_orientation")
    if relational_mood.confidence_signal >= 0.46 and social_tension < 0.52:
        pause_mode = "confident_brief"
        cues.append("confidence_stability")

    cues.extend(
        cue
        for cue in (
            f"gaze_{gaze_mode}",
            f"pause_{pause_mode}",
            f"proximity_{proximity_mode}",
            f"silence_{silence_mode}",
            f"gesture_{gesture_mode}",
        )
        if cue not in cues
    )
    return NonverbalProfile(
        gaze_mode=gaze_mode,
        pause_mode=pause_mode,
        proximity_mode=proximity_mode,
        silence_mode=silence_mode,
        gesture_mode=gesture_mode,
        cues=cues,
    )
