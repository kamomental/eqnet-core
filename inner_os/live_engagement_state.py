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


def _winner_and_margin(scores: Mapping[str, float]) -> tuple[str, float]:
    ordered = sorted(
        ((str(key), _float01(value)) for key, value in dict(scores).items()),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )
    if not ordered:
        return "hold", 0.0
    winner = ordered[0][0]
    top = ordered[0][1]
    runner = ordered[1][1] if len(ordered) > 1 else 0.0
    return winner, max(0.0, min(1.0, top - runner))


@dataclass(frozen=True)
class LiveEngagementState:
    state: str
    score: float
    scores: dict[str, float]
    winner_margin: float
    comment_pickup_room: float
    riff_room: float
    topic_seed_room: float
    audience_address_room: float
    primary_move: str
    dominant_inputs: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "score": round(self.score, 4),
            "scores": {
                str(key): round(float(value), 4)
                for key, value in self.scores.items()
            },
            "winner_margin": round(self.winner_margin, 4),
            "comment_pickup_room": round(self.comment_pickup_room, 4),
            "riff_room": round(self.riff_room, 4),
            "topic_seed_room": round(self.topic_seed_room, 4),
            "audience_address_room": round(self.audience_address_room, 4),
            "primary_move": self.primary_move,
            "dominant_inputs": list(self.dominant_inputs),
        }


def derive_live_engagement_state(
    *,
    self_state: Mapping[str, Any],
    initiative_readiness: Mapping[str, Any],
    initiative_followup_bias: Mapping[str, Any],
    relational_style_memory_state: Mapping[str, Any],
    lightness_budget_state: Mapping[str, Any],
    social_topology_state: Mapping[str, Any],
    body_recovery_guard: Mapping[str, Any],
    body_homeostasis_state: Mapping[str, Any],
    homeostasis_budget_state: Mapping[str, Any],
    shared_moment_state: Mapping[str, Any] | None = None,
    joint_state: Mapping[str, Any] | None = None,
) -> LiveEngagementState:
    state = dict(self_state or {})
    readiness = dict(initiative_readiness or {})
    followup = dict(initiative_followup_bias or {})
    relation_style = dict(relational_style_memory_state or {})
    lightness_budget = dict(lightness_budget_state or {})
    topology = dict(social_topology_state or {})
    body_guard = dict(body_recovery_guard or {})
    body_homeostasis = dict(body_homeostasis_state or {})
    homeostasis_budget = dict(homeostasis_budget_state or {})
    shared_moment = dict(shared_moment_state or {})
    joint = dict(joint_state or {})

    surface_mode = _text(state.get("interaction_mode") or state.get("mode")).lower()
    talk_mode = _text(state.get("talk_mode")).lower()
    surface_user_text = _text(state.get("surface_user_text"))
    topology_name = _text(topology.get("state"))
    topology_score = _float01(topology.get("score"))
    readiness_state = _text(readiness.get("state"))
    readiness_score = _float01(readiness.get("score"))
    followup_state = _text(followup.get("state"))
    followup_score = _float01(followup.get("score"))
    banter_room = _float01(relation_style.get("banter_room"))
    playful_ceiling = _float01(relation_style.get("playful_ceiling"))
    lexical_variation_bias = _float01(relation_style.get("lexical_variation_bias"))
    lightness_room = _float01(lightness_budget.get("banter_room"))
    guard_state = _text(body_guard.get("state"))
    guard_score = _float01(body_guard.get("score"))
    body_state = _text(body_homeostasis.get("state"))
    body_score = _float01(body_homeostasis.get("score"))
    budget_state = _text(homeostasis_budget.get("state"))
    budget_score = _float01(homeostasis_budget.get("score"))
    shared_moment_name = _text(shared_moment.get("state"))
    shared_moment_kind = _text(shared_moment.get("moment_kind"))
    shared_moment_score = _float01(shared_moment.get("score"))
    shared_moment_jointness = _float01(shared_moment.get("jointness"))
    shared_moment_afterglow = _float01(shared_moment.get("afterglow"))
    joint_mode = _text(joint.get("dominant_mode"))
    joint_shared_delight = _float01(joint.get("shared_delight"))
    joint_shared_tension = _float01(joint.get("shared_tension"))
    joint_common_ground = _float01(joint.get("common_ground"))
    joint_attention = _float01(joint.get("joint_attention"))
    joint_mutual_room = _float01(joint.get("mutual_room"))
    joint_coupling_strength = _float01(joint.get("coupling_strength"))

    streaming_active = 1.0 if surface_mode == "streaming" else 0.0
    followup_open = followup_score if followup_state in {"reopen_softly", "offer_next_step"} else 0.0
    forward_drive = followup_score if followup_state == "offer_next_step" else 0.0
    guard_penalty = max(
        guard_score,
        body_score if body_state in {"depleted", "recovering"} else 0.0,
        budget_score if budget_state == "depleted" else 0.0,
    )
    shared_moment_room = _float01(
        shared_moment_score * 0.58
        + shared_moment_jointness * 0.24
        + shared_moment_afterglow * 0.18
    )
    joint_room = _float01(
        joint_shared_delight * 0.24
        + joint_common_ground * 0.2
        + joint_attention * 0.2
        + joint_mutual_room * 0.18
        + joint_coupling_strength * 0.18
        - joint_shared_tension * 0.14
    )
    engagement_talk_mode = talk_mode
    if (
        engagement_talk_mode in {"", "watch"}
        and surface_user_text
        and shared_moment_name == "shared_moment"
        and lightness_room >= 0.24
        and guard_penalty <= 0.32
        and shared_moment_room >= 0.24
    ):
        engagement_talk_mode = "talk"
    talk_drive = {
        "talk": 1.0,
        "ask": 0.72,
        "watch": 0.24,
        "presence": 0.08,
        "soothe": 0.18,
    }.get(engagement_talk_mode, 0.18)
    audience_address_room = _float01(
        0.42 * streaming_active
        + 0.18 * talk_drive
        + 0.12 * topology_score
        + 0.1 * lightness_room
        + 0.08 * joint_room
        - 0.14 * (1.0 if topology_name == "hierarchical" else 0.0)
        - 0.08 * (1.0 if topology_name == "one_to_one" else 0.0)
    )
    joint_pickup_bonus = _float01(
        joint_attention * 0.22
        + joint_mutual_room * 0.18
        + joint_coupling_strength * 0.18
        + joint_common_ground * 0.16
        + joint_shared_delight * 0.12
        + (0.08 if joint_mode in {"delighted_jointness", "shared_attention"} else 0.0)
        - joint_shared_tension * 0.18
    )
    shared_moment_pickup_bonus = _float01(
        (0.16 if shared_moment_name == "shared_moment" else 0.0)
        + (0.06 if shared_moment_kind in {"laugh", "relief"} else 0.0)
        + (0.08 if talk_mode in {"talk", "ask"} else 0.0)
        + (0.06 if lightness_room >= 0.24 else 0.0)
        - 0.22 * guard_penalty
        - 0.06 * (1.0 if topology_name == "hierarchical" else 0.0)
    )

    comment_pickup_room = _float01(
        0.34 * streaming_active
        + 0.18 * talk_drive
        + 0.16 * readiness_score
        + 0.12 * followup_open
        + 0.08 * audience_address_room
        + 0.06 * lightness_room
        + 0.18 * shared_moment_room
        + 0.14 * joint_room
        + 0.12 * joint_pickup_bonus
        + 0.14 * shared_moment_pickup_bonus
        - 0.18 * guard_penalty
    )
    riff_room = _float01(
        0.3 * streaming_active
        + 0.2 * (1.0 if engagement_talk_mode == "talk" else 0.0)
        + 0.14 * readiness_score
        + 0.1 * followup_open
        + 0.12 * banter_room
        + 0.08 * playful_ceiling
        + 0.06 * lexical_variation_bias
        + 0.06 * lightness_room
        + 0.16 * shared_moment_room
        + 0.14 * joint_room
        + 0.1 * joint_pickup_bonus
        + 0.12 * shared_moment_pickup_bonus
        - 0.2 * guard_penalty
        - 0.08 * (1.0 if topology_name == "hierarchical" else 0.0)
    )
    topic_seed_room = _float01(
        0.3 * streaming_active
        + 0.24 * (1.0 if engagement_talk_mode == "talk" else 0.0)
        + 0.18 * readiness_score
        + 0.14 * forward_drive
        + 0.08 * audience_address_room
        + 0.08 * lightness_room
        + 0.08 * shared_moment_room
        + 0.1 * joint_room
        + 0.08 * joint_pickup_bonus
        + 0.08 * shared_moment_pickup_bonus
        - 0.22 * guard_penalty
        - 0.06 * (1.0 if topology_name == "hierarchical" else 0.0)
    )

    hold_score = _float01(
        0.46 * (1.0 - streaming_active)
        + 0.16 * (1.0 - talk_drive)
        + 0.18 * guard_score
        + 0.12 * (body_score if body_state in {"depleted", "recovering"} else 0.0)
        + 0.1 * (budget_score if budget_state == "depleted" else 0.0)
        + 0.08 * (1.0 if topology_name == "hierarchical" else 0.0)
        + 0.06 * (1.0 if readiness_state == "hold" else 0.0)
        - 0.2 * shared_moment_room
        - 0.14 * joint_room
        - 0.12 * joint_pickup_bonus
        - 0.14 * shared_moment_pickup_bonus
    )
    pickup_score = _float01(
        comment_pickup_room
        + (0.08 if engagement_talk_mode == "ask" else 0.0)
        - 0.08 * hold_score
    )
    riff_score = _float01(
        riff_room
        + (0.06 if engagement_talk_mode == "talk" else 0.0)
        - 0.08 * hold_score
    )
    seed_score = _float01(
        topic_seed_room
        + (0.08 if readiness_state == "ready" else 0.0)
        - 0.1 * hold_score
    )
    scores = {
        "hold": hold_score,
        "pickup_comment": pickup_score,
        "riff_with_comment": riff_score,
        "seed_topic": seed_score,
    }
    winner, winner_margin = _winner_and_margin(scores)
    primary_move = {
        "pickup_comment": "pick_up_comment",
        "riff_with_comment": "riff_current_comment",
        "seed_topic": "seed_small_topic",
    }.get(winner, "hold_presence")
    dominant_inputs = [
        label
        for label, enabled in (
            ("streaming_mode", streaming_active >= 1.0),
            ("talk_mode_talk", engagement_talk_mode == "talk"),
            ("talk_mode_ask", engagement_talk_mode == "ask"),
            (
                "shared_moment_talk_override",
                engagement_talk_mode == "talk" and talk_mode in {"", "watch"},
            ),
            ("initiative_ready", readiness_state == "ready" and readiness_score >= 0.42),
            ("followup_offer_next_step", followup_state == "offer_next_step" and followup_score >= 0.22),
            ("followup_reopen_softly", followup_state == "reopen_softly" and followup_score >= 0.22),
            ("relational_banter_room", banter_room >= 0.28),
            ("lightness_banter_room", lightness_room >= 0.24),
            ("shared_moment", shared_moment_name == "shared_moment" and shared_moment_room >= 0.22),
            (f"shared_moment_{shared_moment_kind}", shared_moment_kind in {"laugh", "relief", "pleasant_surprise"}),
            ("shared_moment_pickup_bonus", shared_moment_pickup_bonus >= 0.12),
            ("joint_room_open", joint_room >= 0.24),
            ("joint_pickup_bonus", joint_pickup_bonus >= 0.14),
            (f"joint_mode_{joint_mode}", joint_mode in {"delighted_jointness", "shared_attention", "repair_attunement"}),
            ("audience_address_room", audience_address_room >= 0.28),
            ("body_guard", guard_state in {"guarded", "recovery_first"}),
            ("body_homeostasis_depleted", body_state == "depleted"),
            ("homeostasis_budget_depleted", budget_state == "depleted"),
            ("hierarchical_topology", topology_name == "hierarchical" and topology_score >= 0.34),
        )
        if enabled
    ]
    return LiveEngagementState(
        state=winner,
        score=scores[winner],
        scores=scores,
        winner_margin=winner_margin,
        comment_pickup_room=comment_pickup_room,
        riff_room=riff_room,
        topic_seed_room=topic_seed_room,
        audience_address_room=audience_address_room,
        primary_move=primary_move,
        dominant_inputs=dominant_inputs,
    )
