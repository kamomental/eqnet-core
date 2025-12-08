"""Style modulation utilities."""

from __future__ import annotations

from typing import Any, Dict

from .style_state import UtteranceStyleState


def modulate_style(
    base_style: UtteranceStyleState,
    metrics: Dict[str, Any],
    context: Dict[str, Any],
) -> UtteranceStyleState:
    """Return a style state adjusted by current metrics/context."""

    style_dict = base_style.__dict__.copy()

    hr_emo = float(metrics.get("heart_rate_emotion", 0.0) or 0.0)
    inner_emo = float(metrics.get("inner_emotion_score", 0.0) or 0.0)
    body_flag = str(metrics.get("body_state_flag", "normal") or "normal")
    mode = context.get("mode", "default")

    mood = style_dict.get("mood", "calm")
    filler_level = float(style_dict.get("filler_level", 0.3))
    colloquial_level = float(style_dict.get("colloquial_level", 0.3))

    if hr_emo < 0.5 and inner_emo < 0.5:
        mood = "calm"
    elif hr_emo > 1.5 or inner_emo > 1.2:
        mood = "excited"
        filler_level = min(1.0, filler_level + 0.1)
        colloquial_level = min(1.0, colloquial_level + 0.1)
    else:
        mood = "happy"

    if body_flag == "overloaded":
        mood = "tired"
        filler_level *= 0.8
        colloquial_level *= 0.7

    style_dict["mood"] = mood
    style_dict["filler_level"] = filler_level
    style_dict["colloquial_level"] = colloquial_level

    if mode == "business":
        style_dict["politeness"] = "desu_masu"
        style_dict["self_pronoun"] = "私"
        style_dict["other_pronoun"] = "あなた"

    return UtteranceStyleState(**style_dict)
