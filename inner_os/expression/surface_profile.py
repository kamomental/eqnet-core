from __future__ import annotations

from typing import Any, Mapping


def derive_surface_profile(
    *,
    speech_act: str,
    utterance_stance: str,
    orchestration: Mapping[str, Any],
    live_regulation: Any,
    nonverbal_profile: Any,
) -> dict[str, object]:
    contact_readiness = float(orchestration.get("contact_readiness", 0.0) or 0.0)
    coherence_score = float(orchestration.get("coherence_score", 0.0) or 0.0)
    strained_pause = float(getattr(live_regulation, "strained_pause", 0.0) or 0.0)
    future_loop_pull = float(getattr(live_regulation, "future_loop_pull", 0.0) or 0.0)
    fantasy_loop_pull = float(getattr(live_regulation, "fantasy_loop_pull", 0.0) or 0.0)

    opening_delay = "brief"
    if strained_pause >= 0.55 or getattr(nonverbal_profile, "pause_mode", "") in {"measured_ritual", "waiting"}:
        opening_delay = "long"
    elif strained_pause >= 0.34 or getattr(nonverbal_profile, "pause_mode", "") in {"patient_care", "soft"}:
        opening_delay = "measured"

    response_length = "balanced"
    if speech_act == "check_in" and contact_readiness >= 0.52 and coherence_score >= 0.42:
        response_length = "short"
    if fantasy_loop_pull >= 0.42:
        response_length = "reflective"
    elif future_loop_pull >= 0.5 and contact_readiness >= 0.48:
        response_length = "forward_leaning"

    sentence_temperature = "neutral"
    if utterance_stance == "warm_check_in":
        sentence_temperature = "warm"
    elif utterance_stance == "gentle_check_in":
        sentence_temperature = "gentle"
    elif utterance_stance == "measured_check_in":
        sentence_temperature = "measured"

    pause_insertion = "none"
    if opening_delay == "long":
        pause_insertion = "visible_pause"
    elif opening_delay == "measured":
        pause_insertion = "soft_pause"

    certainty_style = "direct"
    if fantasy_loop_pull >= 0.36:
        certainty_style = "tentative"
    elif strained_pause >= 0.48:
        certainty_style = "careful"

    cues: list[str] = [
        f"surface_delay_{opening_delay}",
        f"surface_length_{response_length}",
        f"surface_temperature_{sentence_temperature}",
        f"surface_certainty_{certainty_style}",
    ]
    if pause_insertion != "none":
        cues.append(f"surface_pause_{pause_insertion}")

    return {
        "opening_delay": opening_delay,
        "response_length": response_length,
        "sentence_temperature": sentence_temperature,
        "pause_insertion": pause_insertion,
        "certainty_style": certainty_style,
        "cues": cues,
    }
