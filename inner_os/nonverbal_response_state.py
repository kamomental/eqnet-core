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
class NonverbalResponseState:
    state: str
    score: float
    response_kind: str
    pause_mode: str
    silence_mode: str
    token_profile: str
    nod_mode: str
    breath_mode: str
    timing_bias: str
    dominant_inputs: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "score": round(self.score, 4),
            "response_kind": self.response_kind,
            "pause_mode": self.pause_mode,
            "silence_mode": self.silence_mode,
            "token_profile": self.token_profile,
            "nod_mode": self.nod_mode,
            "breath_mode": self.breath_mode,
            "timing_bias": self.timing_bias,
            "dominant_inputs": list(self.dominant_inputs),
        }


def derive_nonverbal_response_state(
    *,
    listener_action_state: Mapping[str, Any],
    presence_hold_state: Mapping[str, Any],
    shared_moment_state: Mapping[str, Any] | None = None,
    live_engagement_state: Mapping[str, Any] | None = None,
    utterance_reason_packet: Mapping[str, Any] | None = None,
    terrain_dynamics_state: Mapping[str, Any] | None = None,
    joint_state: Mapping[str, Any] | None = None,
) -> NonverbalResponseState:
    listener = dict(listener_action_state or {})
    presence = dict(presence_hold_state or {})
    shared_moment = dict(shared_moment_state or {})
    live = dict(live_engagement_state or {})
    utterance_reason = dict(utterance_reason_packet or {})
    terrain = dict(terrain_dynamics_state or {})
    joint = dict(joint_state or {})

    listener_name = _text(listener.get("state"))
    listener_score = _float01(listener.get("score"))
    token_profile = _text(listener.get("token_profile")) or "plain_ack"
    filler_mode = _text(listener.get("filler_mode"))
    laughter_room = _float01(listener.get("laughter_room"))
    acknowledgement_room = _float01(listener.get("acknowledgement_room"))
    presence_name = _text(presence.get("state"))
    presence_score = _float01(presence.get("score"))
    presence_silence_mode = _text(presence.get("silence_mode")) or "neutral"
    shared_moment_name = _text(shared_moment.get("state"))
    shared_moment_kind = _text(shared_moment.get("moment_kind"))
    shared_moment_score = _float01(shared_moment.get("score"))
    live_name = _text(live.get("state"))
    utterance_reason_state = _text(utterance_reason.get("state"))
    utterance_reason_relation_frame = _text(utterance_reason.get("relation_frame"))
    utterance_reason_causal_frame = _text(utterance_reason.get("causal_frame"))
    utterance_reason_memory_frame = _text(utterance_reason.get("memory_frame"))
    utterance_reason_preserve = _text(utterance_reason.get("preserve"))
    joint_shared_delight = _float01(joint.get("shared_delight"))
    joint_shared_tension = _float01(joint.get("shared_tension"))
    terrain_barrier_height = _float01(terrain.get("barrier_height"))
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

    state = "quiet_presence"
    response_kind = "hold"
    pause_mode = "patient_care" if presence_name in {"holding_space", "backchannel_ready_hold"} else "neutral"
    silence_mode = presence_silence_mode
    nod_mode = "still"
    breath_mode = "settle"
    timing_bias = "wait"
    dominant_inputs: list[str] = []
    score = max(listener_score, presence_score)

    if (
        shared_moment_name == "shared_moment"
        and shared_moment_kind == "laugh"
        and listener_name == "warm_laugh_ack"
        and presence_name in {"backchannel_ready_hold", "reentry_open", "steady_presence"}
    ):
        state = "warm_laugh_ack"
        response_kind = "backchannel"
        pause_mode = "short_warm"
        silence_mode = "shared"
        nod_mode = "small_sync"
        breath_mode = "light_release"
        timing_bias = "near_turn_end"
        dominant_inputs.extend(["shared_moment:laugh", "listener:warm_laugh_ack"])
        score = max(score, laughter_room, shared_moment_score)
    elif relation_reentry_signal and presence_name in {"reentry_open", "steady_presence"} and terrain_barrier_height <= 0.42:
        state = "bridge_ack_presence"
        response_kind = "backchannel"
        pause_mode = "soft"
        silence_mode = "soft_release"
        nod_mode = "micro_return"
        breath_mode = "ready"
        timing_bias = "just_after_turn"
        dominant_inputs.append("reason:relation_reentry")
        score = max(score, acknowledgement_room, presence_score)
    elif acknowledgement_room >= 0.36 and presence_name in {"backchannel_ready_hold", "reentry_open"}:
        state = "soft_ack_presence"
        response_kind = "backchannel"
        pause_mode = "soft"
        silence_mode = "shared_hold" if presence_name == "backchannel_ready_hold" else "soft_release"
        nod_mode = "small_sync"
        breath_mode = "steady"
        timing_bias = "just_after_turn"
        dominant_inputs.append("listener:ack")
        score = max(score, acknowledgement_room)
    elif guarded_relation_signal and presence_name in {"holding_space", "backchannel_ready_hold", "steady_presence"}:
        state = "guarded_hold_presence"
        response_kind = "hold"
        pause_mode = "patient_care"
        silence_mode = "protective_hold"
        nod_mode = "still"
        breath_mode = "settle"
        timing_bias = "wait"
        dominant_inputs.append("reason:guarded_relation")
        score = max(score, presence_score, joint_shared_tension)
    elif presence_name == "reentry_open" and live_name in {"pickup_comment", "riff_with_comment", "seed_topic"}:
        state = "lead_in_presence"
        response_kind = "speak_lead_in"
        pause_mode = "confident_brief" if joint_shared_delight >= joint_shared_tension else "soft"
        silence_mode = "soft_release"
        nod_mode = "micro_return"
        breath_mode = "ready"
        timing_bias = "turn_entry"
        dominant_inputs.append("presence:reentry_open")
        score = max(score, presence_score, joint_shared_delight)
    else:
        dominant_inputs.append(f"presence:{presence_name or 'neutral'}")

    if filler_mode:
        dominant_inputs.append(f"filler:{filler_mode}")
    if live_name:
        dominant_inputs.append(f"live:{live_name}")

    return NonverbalResponseState(
        state=state,
        score=score,
        response_kind=response_kind,
        pause_mode=pause_mode,
        silence_mode=silence_mode,
        token_profile=token_profile,
        nod_mode=nod_mode,
        breath_mode=breath_mode,
        timing_bias=timing_bias,
        dominant_inputs=dominant_inputs,
    )
