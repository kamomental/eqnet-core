from __future__ import annotations

from typing import Any, Mapping

from .models import InteractionTrace


def summarize_interaction_trace(
    *,
    sensor_input: Mapping[str, Any],
    current_state: Mapping[str, Any] | None = None,
) -> InteractionTrace:
    observed_gaze_mode = _text(sensor_input.get("observed_gaze_mode")) or _text((current_state or {}).get("observed_gaze_mode"))
    observed_pause_mode = _text(sensor_input.get("observed_pause_mode")) or _text((current_state or {}).get("observed_pause_mode"))
    observed_proximity_mode = _text(sensor_input.get("observed_proximity_mode")) or _text((current_state or {}).get("observed_proximity_mode"))
    observed_hesitation_tone = _text(sensor_input.get("observed_hesitation_tone")) or _text((current_state or {}).get("observed_hesitation_tone"))

    mutual_attention_score = _clamp01(_float(sensor_input.get("mutual_attention_score"), _float((current_state or {}).get("mutual_attention_score"), 0.0)))
    gaze_hold_ratio = _clamp01(_float(sensor_input.get("gaze_hold_ratio"), _float((current_state or {}).get("gaze_hold_ratio"), 0.0)))
    gaze_aversion_ratio = _clamp01(_float(sensor_input.get("gaze_aversion_ratio"), _float((current_state or {}).get("gaze_aversion_ratio"), 0.0)))
    pause_latency = _clamp01(_float(sensor_input.get("pause_latency"), _float((current_state or {}).get("pause_latency"), 0.0)))
    hesitation_signal = _clamp01(_float(sensor_input.get("hesitation_signal"), _float((current_state or {}).get("hesitation_signal"), 0.0)))
    proximity_delta = _float(sensor_input.get("proximity_delta"), _float((current_state or {}).get("proximity_delta"), 0.0))
    repair_signal = _clamp01(_float(sensor_input.get("repair_signal"), _float((current_state or {}).get("repair_signal"), 0.0)))

    cues: list[str] = []

    gaze_mode = observed_gaze_mode or "steady"
    if not observed_gaze_mode:
        if mutual_attention_score >= 0.62:
            gaze_mode = "shared_attention_hold"
        elif gaze_aversion_ratio >= 0.58:
            gaze_mode = "avert"
        elif repair_signal >= 0.45:
            gaze_mode = "gentle_return"
        elif gaze_hold_ratio >= 0.5:
            gaze_mode = "soft_hold"

    pause_mode = observed_pause_mode or "neutral"
    if not observed_pause_mode:
        if hesitation_signal >= 0.66 or pause_latency >= 0.72:
            pause_mode = "measured_ritual"
        elif repair_signal >= 0.5:
            pause_mode = "patient_care"
        elif pause_latency >= 0.46:
            pause_mode = "waiting"
        elif mutual_attention_score >= 0.5 and hesitation_signal <= 0.28:
            pause_mode = "confident_brief"

    proximity_mode = observed_proximity_mode or "neutral"
    if not observed_proximity_mode:
        if proximity_delta >= 0.18:
            proximity_mode = "gentle_near"
        elif proximity_delta <= -0.18:
            proximity_mode = "holding_space"
        elif repair_signal >= 0.5:
            proximity_mode = "holding_space"

    hesitation_tone = observed_hesitation_tone or pause_mode

    if mutual_attention_score >= 0.5:
        cues.append("shared_attention_detected")
    if gaze_aversion_ratio >= 0.5:
        cues.append("gaze_aversion_detected")
    if hesitation_signal >= 0.5:
        cues.append("hesitation_detected")
    if repair_signal >= 0.45:
        cues.append("repair_window_detected")
    cues.extend(
        cue
        for cue in (
            f"trace_gaze_{gaze_mode}",
            f"trace_pause_{pause_mode}",
            f"trace_proximity_{proximity_mode}",
        )
        if cue not in cues
    )

    return InteractionTrace(
        gaze_mode=gaze_mode,
        pause_mode=pause_mode,
        proximity_mode=proximity_mode,
        hesitation_tone=hesitation_tone,
        shared_attention=mutual_attention_score,
        repair_signal=repair_signal,
        cues=cues,
    )


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
