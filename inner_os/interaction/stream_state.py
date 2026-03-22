from __future__ import annotations

from typing import Any, Mapping

from .models import InteractionStreamState, InteractionTrace


def advance_interaction_stream(
    *,
    orchestration: Mapping[str, Any],
    interaction_trace: InteractionTrace,
    previous: Mapping[str, Any] | None = None,
) -> InteractionStreamState:
    prev_shared = _float((previous or {}).get("shared_attention_level"), _float(orchestration.get("coherence_score"), 0.0))
    prev_pause = _float((previous or {}).get("strained_pause_level"), 0.0)
    prev_repair_hold = _float((previous or {}).get("repair_window_hold"), 0.0)
    prev_contact = _float((previous or {}).get("contact_readiness"), _float(orchestration.get("contact_readiness"), 0.0))
    prev_presence = _float((previous or {}).get("human_presence_signal"), _float(orchestration.get("human_presence_signal"), 0.0))
    prev_shared_window = _float_list((previous or {}).get("shared_attention_window"))
    prev_pause_window = _float_list((previous or {}).get("strained_pause_window"))
    prev_updates = int((previous or {}).get("update_count", 0) or 0)

    shared_attention_level = _clamp01(prev_shared * 0.46 + interaction_trace.shared_attention * 0.54)
    strained_pause_level = _clamp01(
        prev_pause * 0.44
        + (0.62 if interaction_trace.pause_mode in {"measured_ritual", "waiting"} else 0.18) * 0.36
        + interaction_trace.repair_signal * 0.2
    )
    shared_attention_window = _window_push(prev_shared_window, shared_attention_level)
    strained_pause_window = _window_push(prev_pause_window, strained_pause_level)
    shared_attention_mean = _mean(shared_attention_window)
    strained_pause_mean = _mean(strained_pause_window)
    repair_window_hold = _clamp01(
        prev_repair_hold * 0.62
        + interaction_trace.repair_signal * 0.28
        + (0.18 if bool(orchestration.get("repair_bias")) else 0.0)
    )
    repair_window_open = bool(
        repair_window_hold >= 0.36
        or interaction_trace.repair_signal >= 0.45
        or strained_pause_mean >= 0.56
    )
    contact_readiness = _clamp01(
        prev_contact * 0.42
        + _float(orchestration.get("contact_readiness"), 0.0) * 0.28
        + shared_attention_mean * 0.2
        - strained_pause_mean * 0.12
        + (0.08 if interaction_trace.proximity_mode in {"gentle_near", "future_opening"} else 0.0)
    )
    human_presence_signal = _clamp01(
        prev_presence * 0.4
        + _float(orchestration.get("human_presence_signal"), 0.0) * 0.28
        + shared_attention_mean * 0.18
        + (0.08 if interaction_trace.gaze_mode in {"shared_attention_hold", "soft_hold", "gentle_return"} else 0.0)
        + (0.06 if repair_window_open else 0.0)
    )

    cues: list[str] = []
    if shared_attention_mean >= 0.5:
        cues.append("stream_shared_attention_up")
    if strained_pause_mean >= 0.48:
        cues.append("stream_strained_pause")
    if repair_window_open:
        cues.append("stream_repair_window_open")
    if contact_readiness >= 0.5:
        cues.append("stream_contact_ready")
    if human_presence_signal >= 0.54:
        cues.append("stream_presence_dense")

    return InteractionStreamState(
        shared_attention_level=round(shared_attention_level, 4),
        strained_pause_level=round(strained_pause_level, 4),
        repair_window_open=repair_window_open,
        repair_window_hold=round(repair_window_hold, 4),
        contact_readiness=round(contact_readiness, 4),
        human_presence_signal=round(human_presence_signal, 4),
        shared_attention_window=[round(value, 4) for value in shared_attention_window],
        strained_pause_window=[round(value, 4) for value in strained_pause_window],
        update_count=prev_updates + 1,
        cues=cues,
    )


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out[-3:]


def _window_push(window: list[float], value: float) -> list[float]:
    return [*window[-2:], float(value)]


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
