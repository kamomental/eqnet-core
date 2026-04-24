from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _float01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


@dataclass(frozen=True)
class ResponseSelectionState:
    state: str
    channel: str
    score: float
    speak_room: float
    backchannel_room: float
    hold_room: float
    defer_room: float
    selected_signal: str
    dominant_inputs: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "channel": self.channel,
            "score": round(self.score, 4),
            "speak_room": round(self.speak_room, 4),
            "backchannel_room": round(self.backchannel_room, 4),
            "hold_room": round(self.hold_room, 4),
            "defer_room": round(self.defer_room, 4),
            "selected_signal": self.selected_signal,
            "dominant_inputs": list(self.dominant_inputs),
        }


def derive_response_selection_state(
    *,
    primary_action: str,
    execution_mode: str,
    reply_permission: str,
    defer_dominance: float,
    live_engagement_state: Mapping[str, Any],
    presence_hold_state: Mapping[str, Any],
    nonverbal_response_state: Mapping[str, Any],
    shared_moment_state: Mapping[str, Any] | None = None,
    utterance_reason_packet: Mapping[str, Any] | None = None,
    subjective_scene_state: Mapping[str, Any] | None = None,
    self_other_attribution_state: Mapping[str, Any] | None = None,
    shared_presence_state: Mapping[str, Any] | None = None,
) -> ResponseSelectionState:
    live = dict(live_engagement_state or {})
    presence = dict(presence_hold_state or {})
    nonverbal = dict(nonverbal_response_state or {})
    shared_moment = dict(shared_moment_state or {})
    utterance_reason = dict(utterance_reason_packet or {})
    subjective_scene = dict(subjective_scene_state or {})
    attribution = dict(self_other_attribution_state or {})
    shared_presence = dict(shared_presence_state or {})

    live_name = _text(live.get("state"))
    live_score = _float01(live.get("score"))
    presence_name = _text(presence.get("state"))
    presence_hold_room = _float01(presence.get("hold_room"))
    presence_backchannel_room = _float01(presence.get("backchannel_room"))
    presence_release_readiness = _float01(presence.get("release_readiness"))
    nonverbal_kind = _text(nonverbal.get("response_kind"))
    nonverbal_score = _float01(nonverbal.get("score"))
    shared_moment_name = _text(shared_moment.get("state"))
    shared_moment_score = _float01(shared_moment.get("score"))
    utterance_reason_state = _text(utterance_reason.get("state"))
    utterance_reason_relation_frame = _text(utterance_reason.get("relation_frame"))
    utterance_reason_causal_frame = _text(utterance_reason.get("causal_frame"))
    utterance_reason_memory_frame = _text(utterance_reason.get("memory_frame"))
    utterance_reason_preserve = _text(utterance_reason.get("preserve"))
    subjective_anchor_frame = _text(subjective_scene.get("anchor_frame"))
    subjective_shared_scene_potential = _float01(subjective_scene.get("shared_scene_potential"))
    attribution_dominant = _text(attribution.get("dominant_attribution"))
    attribution_unknown = _float01(attribution.get("unknown_likelihood"))
    shared_presence_mode = _text(shared_presence.get("dominant_mode"))
    shared_presence_co_presence = _float01(shared_presence.get("co_presence"))
    shared_presence_boundary_stability = _float01(shared_presence.get("boundary_stability"))
    shared_presence_join_signal = max(
        subjective_shared_scene_potential,
        shared_presence_co_presence,
    )
    guarded_self_view_signal = max(
        attribution_unknown,
        max(0.0, 1.0 - shared_presence_boundary_stability),
    )

    relation_reentry_signal = (
        utterance_reason_state == "active"
        and (
            utterance_reason_relation_frame in {"cross_context_bridge", "returning_pattern"}
            or utterance_reason_causal_frame in {"reframing_cause", "memory_trigger_cause"}
            or utterance_reason_memory_frame
            in {
                "echo_known_thread",
                "name_small_return",
                "name_distant_link",
                "echo_returning_pattern",
            }
        )
    )
    guarded_relation_signal = (
        utterance_reason_state == "active"
        and (
            utterance_reason_relation_frame == "unfinished_link"
            or utterance_reason_causal_frame in {"unfinished_thread_cause", "memory_trigger_cause"}
            or utterance_reason_memory_frame
            in {"keep_unfinished_link_near", "keep_known_thread_near", "echo_known_thread"}
        )
        and utterance_reason_preserve in {"", "do_not_overclaim"}
    )

    speak_room = _float01(
        (0.22 if primary_action not in {"hold_presence", "defer_contact", "leave_return_point"} else 0.06)
        + (
            0.22
            if primary_action in {"riff_current_comment", "pick_up_comment", "seed_small_topic", "co_move"}
            else 0.0
        )
        + live_score * 0.3
        + presence_release_readiness * 0.14
        + (0.12 if nonverbal_kind == "speak_lead_in" else 0.0)
        + (0.08 if execution_mode in {"shared_progression", "attuned_contact"} else 0.0)
        + shared_moment_score * 0.08
        + (0.08 if relation_reentry_signal and execution_mode in {"shared_progression", "attuned_contact"} else 0.0)
        - presence_hold_room * 0.12
        - (0.08 if guarded_relation_signal else 0.0)
        + (
            0.1
            if attribution_dominant == "shared"
            and shared_presence_join_signal >= 0.56
            and shared_presence_boundary_stability >= 0.42
            else 0.0
        )
        + (0.04 if subjective_anchor_frame in {"shared_margin", "front_field"} else 0.0)
        - (0.22 if guarded_self_view_signal >= 0.58 else 0.0)
    )
    backchannel_room = _float01(
        presence_backchannel_room * 0.42
        + nonverbal_score * 0.32
        + (0.14 if nonverbal_kind == "backchannel" else 0.0)
        + (0.08 if shared_moment_name == "shared_moment" else 0.0)
        + (0.1 if relation_reentry_signal else 0.0)
        - defer_dominance * 0.08
        - (0.06 if guarded_relation_signal else 0.0)
        + (
            0.1
            if attribution_dominant == "shared"
            and shared_presence_join_signal >= 0.52
            and shared_presence_mode in {"inhabited_shared_space", "soft_projection"}
            else 0.0
        )
        - (0.08 if guarded_self_view_signal >= 0.56 else 0.0)
    )
    hold_room = _float01(
        presence_hold_room * 0.56
        + (0.16 if primary_action == "hold_presence" else 0.0)
        + (0.08 if nonverbal_kind == "hold" else 0.0)
        + (0.08 if presence_name == "holding_space" else 0.0)
        + (0.14 if guarded_relation_signal else 0.0)
        - presence_release_readiness * 0.12
        - (0.08 if relation_reentry_signal else 0.0)
        + (
            0.22
            if guarded_self_view_signal >= 0.56
            or shared_presence_mode == "guarded_boundary"
            else 0.0
        )
    )
    defer_room = _float01(
        defer_dominance * 0.54
        + (0.18 if reply_permission in {"hold_or_brief", "defer"} else 0.0)
        + (0.1 if primary_action in {"defer_contact", "leave_return_point"} else 0.0)
        - live_score * 0.08
        + (0.08 if guarded_relation_signal else 0.0)
        - (0.06 if relation_reentry_signal else 0.0)
        + (0.12 if guarded_self_view_signal >= 0.62 else 0.0)
    )

    channel = "speak"
    state = "speak_response"
    selected_signal = "speak_room"
    dominant_inputs: list[str] = []
    score = speak_room

    if defer_room >= max(speak_room, backchannel_room, hold_room) and defer_room >= 0.48:
        channel = "defer"
        state = "defer_response"
        selected_signal = "defer_room"
        score = defer_room
        dominant_inputs.append("defer_dominance")
    elif hold_room >= max(speak_room, backchannel_room) and hold_room >= 0.46:
        channel = "hold"
        state = "hold_response"
        selected_signal = "hold_room"
        score = hold_room
        dominant_inputs.append("presence_hold")
    elif backchannel_room >= max(speak_room, hold_room) and backchannel_room >= 0.42:
        channel = "backchannel"
        state = "backchannel_response"
        selected_signal = "backchannel_room"
        score = backchannel_room
        dominant_inputs.append("nonverbal_backchannel")

    if relation_reentry_signal:
        dominant_inputs.append("reason:relation_reentry")
    if guarded_relation_signal:
        dominant_inputs.append("reason:guarded_relation")
    if attribution_dominant:
        dominant_inputs.append(f"attribution:{attribution_dominant}")
    if shared_presence_mode:
        dominant_inputs.append(f"shared_presence:{shared_presence_mode}")
    if guarded_self_view_signal >= 0.56:
        dominant_inputs.append("reason:self_view_guard")
    elif attribution_dominant == "shared" and shared_presence_join_signal >= 0.52:
        dominant_inputs.append("reason:shared_presence_join")
    if live_name:
        dominant_inputs.append(f"live:{live_name}")
    if presence_name:
        dominant_inputs.append(f"presence:{presence_name}")
    if nonverbal_kind:
        dominant_inputs.append(f"nonverbal:{nonverbal_kind}")

    return ResponseSelectionState(
        state=state,
        channel=channel,
        score=score,
        speak_room=speak_room,
        backchannel_room=backchannel_room,
        hold_room=hold_room,
        defer_room=defer_room,
        selected_signal=selected_signal,
        dominant_inputs=dominant_inputs,
    )
