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
class PresenceHoldState:
    state: str
    score: float
    hold_room: float
    reentry_room: float
    backchannel_room: float
    release_readiness: float
    silence_mode: str
    pacing_mode: str
    dominant_inputs: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "score": round(self.score, 4),
            "hold_room": round(self.hold_room, 4),
            "reentry_room": round(self.reentry_room, 4),
            "backchannel_room": round(self.backchannel_room, 4),
            "release_readiness": round(self.release_readiness, 4),
            "silence_mode": self.silence_mode,
            "pacing_mode": self.pacing_mode,
            "dominant_inputs": list(self.dominant_inputs),
        }


def derive_presence_hold_state(
    *,
    live_engagement_state: Mapping[str, Any],
    listener_action_state: Mapping[str, Any],
    shared_moment_state: Mapping[str, Any] | None = None,
    utterance_reason_packet: Mapping[str, Any] | None = None,
    joint_state: Mapping[str, Any] | None = None,
    organism_state: Mapping[str, Any] | None = None,
    external_field_state: Mapping[str, Any] | None = None,
    terrain_dynamics_state: Mapping[str, Any] | None = None,
) -> PresenceHoldState:
    live = dict(live_engagement_state or {})
    listener = dict(listener_action_state or {})
    shared_moment = dict(shared_moment_state or {})
    utterance_reason = dict(utterance_reason_packet or {})
    joint = dict(joint_state or {})
    organism = dict(organism_state or {})
    external = dict(external_field_state or {})
    terrain = dict(terrain_dynamics_state or {})

    live_name = _text(live.get("state"))
    live_score = _float01(live.get("score"))
    listener_name = _text(listener.get("state"))
    acknowledgement_room = _float01(listener.get("acknowledgement_room"))
    laughter_room = _float01(listener.get("laughter_room"))
    filler_room = _float01(listener.get("filler_room"))
    shared_moment_name = _text(shared_moment.get("state"))
    shared_moment_kind = _text(shared_moment.get("moment_kind"))
    shared_moment_score = _float01(shared_moment.get("score"))
    shared_moment_afterglow = _float01(shared_moment.get("afterglow"))
    utterance_reason_state = _text(utterance_reason.get("state"))
    utterance_reason_relation_frame = _text(utterance_reason.get("relation_frame"))
    utterance_reason_causal_frame = _text(utterance_reason.get("causal_frame"))
    utterance_reason_memory_frame = _text(utterance_reason.get("memory_frame"))
    utterance_reason_preserve = _text(utterance_reason.get("preserve"))
    joint_mode = _text(joint.get("dominant_mode"))
    joint_shared_delight = _float01(joint.get("shared_delight"))
    joint_shared_tension = _float01(joint.get("shared_tension"))
    joint_common_ground = _float01(joint.get("common_ground"))
    joint_mutual_room = _float01(joint.get("mutual_room"))
    joint_coupling_strength = _float01(joint.get("coupling_strength"))
    organism_posture = _text(organism.get("dominant_posture"))
    organism_play_window = _float01(organism.get("play_window"))
    organism_protective_tension = _float01(organism.get("protective_tension"))
    external_continuity_pull = _float01(external.get("continuity_pull"))
    external_safety_envelope = _float01(external.get("safety_envelope"))
    terrain_recovery_gradient = _float01(terrain.get("recovery_gradient"))
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

    hold_room = _float01(
        0.18
        + organism_protective_tension * 0.24
        + joint_shared_tension * 0.2
        + terrain_barrier_height * 0.16
        + external_safety_envelope * 0.14
        + (0.12 if live_name == "hold" else 0.0)
        + (0.12 if guarded_relation_signal else 0.0)
        - joint_shared_delight * 0.12
        - organism_play_window * 0.1
        - (0.08 if relation_reentry_signal else 0.0)
        - shared_moment_score * 0.08
    )
    reentry_room = _float01(
        joint_common_ground * 0.24
        + joint_mutual_room * 0.18
        + joint_coupling_strength * 0.14
        + external_continuity_pull * 0.16
        + terrain_recovery_gradient * 0.14
        + shared_moment_afterglow * 0.08
        + acknowledgement_room * 0.06
        + (0.16 if relation_reentry_signal else 0.0)
        - terrain_barrier_height * 0.1
        - (0.08 if guarded_relation_signal else 0.0)
        - joint_shared_tension * 0.1
    )
    backchannel_room = _float01(
        acknowledgement_room * 0.34
        + laughter_room * 0.26
        + filler_room * 0.14
        + shared_moment_score * 0.12
        + joint_shared_delight * 0.1
        + (0.1 if listener_name == "warm_laugh_ack" else 0.0)
        + (0.08 if relation_reentry_signal else 0.0)
        - organism_protective_tension * 0.08
        - (0.06 if guarded_relation_signal else 0.0)
        - terrain_barrier_height * 0.08
    )
    release_readiness = _float01(
        reentry_room * 0.56
        + backchannel_room * 0.28
        + live_score * 0.08
        - hold_room * 0.12
    )

    state = "steady_presence"
    silence_mode = "neutral"
    pacing_mode = "steady"
    dominant_inputs: list[str] = []

    if hold_room >= 0.56 and backchannel_room >= 0.34:
        state = "backchannel_ready_hold"
        silence_mode = "shared_hold"
        pacing_mode = "patient"
        dominant_inputs.extend(["hold_room", "backchannel_room"])
    elif hold_room >= 0.56:
        state = "holding_space"
        silence_mode = "protective_hold"
        pacing_mode = "extended"
        dominant_inputs.append("hold_room")
    elif relation_reentry_signal and reentry_room >= 0.34 and backchannel_room >= 0.28:
        state = "reentry_open"
        silence_mode = "soft_release"
        pacing_mode = "soft_return"
        dominant_inputs.extend(["relation_reentry_signal", "reentry_room"])
    elif release_readiness >= 0.46:
        state = "reentry_open"
        silence_mode = "soft_release"
        pacing_mode = "soft_return"
        dominant_inputs.append("release_readiness")

    if shared_moment_name == "shared_moment":
        dominant_inputs.append(f"shared_moment:{shared_moment_kind or 'present'}")
    if relation_reentry_signal:
        dominant_inputs.append("reason:relation_reentry")
    if guarded_relation_signal:
        dominant_inputs.append("reason:guarded_relation")
    if joint_mode:
        dominant_inputs.append(f"joint:{joint_mode}")
    if organism_posture:
        dominant_inputs.append(f"organism:{organism_posture}")
    if live_name:
        dominant_inputs.append(f"live:{live_name}")

    score = max(hold_room, release_readiness)
    return PresenceHoldState(
        state=state,
        score=score,
        hold_room=hold_room,
        reentry_room=reentry_room,
        backchannel_room=backchannel_room,
        release_readiness=release_readiness,
        silence_mode=silence_mode,
        pacing_mode=pacing_mode,
        dominant_inputs=dominant_inputs,
    )
