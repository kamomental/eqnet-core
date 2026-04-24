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
class ListenerActionState:
    state: str
    score: float
    acknowledgement_room: float
    laughter_room: float
    filler_room: float
    filler_mode: str
    token_profile: str
    dominant_inputs: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "score": round(self.score, 4),
            "acknowledgement_room": round(self.acknowledgement_room, 4),
            "laughter_room": round(self.laughter_room, 4),
            "filler_room": round(self.filler_room, 4),
            "filler_mode": self.filler_mode,
            "token_profile": self.token_profile,
            "dominant_inputs": list(self.dominant_inputs),
        }


def derive_listener_action_state(
    *,
    expressive_style_state: Mapping[str, Any],
    cultural_conversation_state: Mapping[str, Any],
    live_engagement_state: Mapping[str, Any],
    shared_moment_state: Mapping[str, Any] | None = None,
    recent_dialogue_state: Mapping[str, Any] | None = None,
) -> ListenerActionState:
    expressive = dict(expressive_style_state or {})
    cultural = dict(cultural_conversation_state or {})
    live = dict(live_engagement_state or {})
    shared_moment = dict(shared_moment_state or {})
    recent = dict(recent_dialogue_state or {})

    expressive_name = _text(expressive.get("state"))
    expressive_lightness_room = _float01(expressive.get("lightness_room"))
    culture_name = _text(cultural.get("state"))
    joke_ratio_ceiling = _float01(cultural.get("joke_ratio_ceiling"))
    politeness_pressure = _float01(cultural.get("politeness_pressure"))
    live_name = _text(live.get("state"))
    live_score = _float01(live.get("score"))
    shared_moment_name = _text(shared_moment.get("state"))
    shared_moment_kind = _text(shared_moment.get("moment_kind"))
    shared_moment_score = _float01(shared_moment.get("score"))
    shared_moment_afterglow = _float01(shared_moment.get("afterglow"))
    recent_name = _text(recent.get("state"))

    continuing_thread = recent_name in {
        "continuing_thread",
        "bright_continuity",
        "reopening_thread",
    }
    bright_pickup = live_name in {"pickup_comment", "riff_with_comment", "seed_topic"}
    playful_style = expressive_name in {"light_playful", "gal_bright", "gal_casual"}
    warm_style = expressive_name in {"warm_companion", "grounded_gentle", "quiet_repair"}
    formal_style = expressive_name in {"reverent_measured", "careful_measured"}

    acknowledgement_room = _float01(
        0.18
        + expressive_lightness_room * 0.22
        + live_score * 0.18
        + (0.1 if continuing_thread else 0.0)
        + (0.08 if warm_style else 0.0)
        - politeness_pressure * 0.12
    )
    laughter_room = _float01(
        shared_moment_score * 0.38
        + shared_moment_afterglow * 0.22
        + joke_ratio_ceiling * 0.18
        + expressive_lightness_room * 0.12
        + (0.1 if shared_moment_kind == "laugh" else 0.0)
        + (0.08 if playful_style else 0.0)
        - politeness_pressure * 0.16
    )
    filler_room = _float01(
        acknowledgement_room * 0.44
        + laughter_room * 0.36
        + (0.1 if bright_pickup else 0.0)
        + (0.06 if continuing_thread else 0.0)
        - (0.08 if formal_style or culture_name in {"public_courteous", "hierarchy_respectful"} else 0.0)
    )

    filler_mode = "caregiver"
    if playful_style or culture_name == "casual_shared":
        filler_mode = "playful"
    elif formal_style or culture_name in {
        "careful_polite",
        "public_courteous",
        "hierarchy_respectful",
    }:
        filler_mode = "professional"

    state = "none"
    token_profile = "plain_ack"
    score = filler_room
    dominant_inputs: list[str] = []
    if laughter_room >= 0.42 and shared_moment_name == "shared_moment":
        state = "warm_laugh_ack"
        token_profile = "soft_laugh"
        score = max(filler_room, laughter_room)
        dominant_inputs.extend(["shared_moment", f"shared_moment_{shared_moment_kind}"])
    elif acknowledgement_room >= 0.4 and bright_pickup:
        state = "playful_ack" if playful_style else "soft_ack"
        token_profile = "double_ack" if continuing_thread else "soft_ack"
        score = max(filler_room, acknowledgement_room)
        dominant_inputs.append("live_pickup")
    elif acknowledgement_room >= 0.34 and continuing_thread:
        state = "soft_ack"
        token_profile = "soft_ack"
        score = acknowledgement_room
        dominant_inputs.append("continuing_thread")

    if warm_style:
        dominant_inputs.append("warm_style")
    if playful_style:
        dominant_inputs.append("playful_style")
    if culture_name:
        dominant_inputs.append(f"culture:{culture_name}")
    if live_name:
        dominant_inputs.append(f"live:{live_name}")

    return ListenerActionState(
        state=state,
        score=score,
        acknowledgement_room=acknowledgement_room,
        laughter_room=laughter_room,
        filler_room=filler_room,
        filler_mode=filler_mode,
        token_profile=token_profile,
        dominant_inputs=dominant_inputs,
    )
