from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..anchor_normalization import select_anchor_hint
from .interaction_constraints import (
    InteractionConstraints,
    coerce_interaction_constraints,
)


@dataclass(frozen=True)
class TurnDelta:
    kind: str = ""
    preferred_act: str = ""
    anchor_hint: str = ""
    priority: float = 0.0
    rationale: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "preferred_act": self.preferred_act,
            "anchor_hint": self.anchor_hint,
            "priority": round(float(self.priority), 4),
            "rationale": list(self.rationale),
        }


def derive_turn_delta(
    interaction_policy: Mapping[str, Any] | None,
    *,
    interaction_constraints: Mapping[str, Any] | InteractionConstraints | None = None,
) -> TurnDelta:
    packet = dict(interaction_policy or {})
    if isinstance(interaction_constraints, InteractionConstraints):
        constraints = interaction_constraints
    else:
        constraints = coerce_interaction_constraints(interaction_constraints)

    identity_arc_kind = str(packet.get("identity_arc_kind") or "").strip()
    temporal_membrane_mode = str(
        packet.get("temporal_membrane_mode")
        or packet.get("temporal_membrane_focus")
        or ""
    ).strip()
    agenda_window_state = _nested_state(packet.get("agenda_window_state"))
    response_strategy = str(packet.get("response_strategy") or "").strip()
    recent_dialogue_state = dict(packet.get("recent_dialogue_state") or {})
    recent_dialogue_kind = str(recent_dialogue_state.get("state") or "").strip()
    try:
        recent_thread_carry = float(recent_dialogue_state.get("thread_carry") or 0.0)
    except (TypeError, ValueError):
        recent_thread_carry = 0.0
    try:
        recent_reopen_pressure = float(recent_dialogue_state.get("reopen_pressure") or 0.0)
    except (TypeError, ValueError):
        recent_reopen_pressure = 0.0
    discussion_thread_state = dict(packet.get("discussion_thread_state") or {})
    discussion_thread_kind = str(discussion_thread_state.get("state") or "").strip()
    discussion_thread_anchor = str(discussion_thread_state.get("topic_anchor") or "").strip()
    try:
        discussion_unresolved_pressure = float(
            discussion_thread_state.get("unresolved_pressure") or 0.0
        )
    except (TypeError, ValueError):
        discussion_unresolved_pressure = 0.0
    try:
        discussion_revisit_readiness = float(
            discussion_thread_state.get("revisit_readiness") or 0.0
        )
    except (TypeError, ValueError):
        discussion_revisit_readiness = 0.0
    issue_state = dict(packet.get("issue_state") or {})
    issue_kind = str(issue_state.get("state") or "").strip()
    issue_anchor = str(issue_state.get("issue_anchor") or "").strip()
    try:
        issue_question_pressure = float(issue_state.get("question_pressure") or 0.0)
    except (TypeError, ValueError):
        issue_question_pressure = 0.0
    try:
        issue_pause_readiness = float(issue_state.get("pause_readiness") or 0.0)
    except (TypeError, ValueError):
        issue_pause_readiness = 0.0
    discussion_registry = dict(packet.get("discussion_thread_registry_snapshot") or {})
    dominant_discussion_anchor = str(discussion_registry.get("dominant_anchor") or "").strip()
    recent_dialogue_anchor = str(recent_dialogue_state.get("recent_anchor") or "").strip()
    autobiographical_thread_mode = str(packet.get("autobiographical_thread_mode") or "").strip()
    autobiographical_thread_anchor = str(packet.get("autobiographical_thread_anchor") or "").strip()
    autobiographical_thread_focus = str(packet.get("autobiographical_thread_focus") or "").strip()
    try:
        autobiographical_thread_strength = float(packet.get("autobiographical_thread_strength") or 0.0)
    except (TypeError, ValueError):
        autobiographical_thread_strength = 0.0
    green_kernel_composition = dict(packet.get("green_kernel_composition") or {})
    green_field = dict(green_kernel_composition.get("field") or {})
    try:
        green_affective_charge = float(green_field.get("affective_charge") or 0.0)
    except (TypeError, ValueError):
        green_affective_charge = 0.0
    try:
        green_guardedness = float(green_field.get("guardedness") or 0.0)
    except (TypeError, ValueError):
        green_guardedness = 0.0
    try:
        green_reopening_pull = float(green_field.get("reopening_pull") or 0.0)
    except (TypeError, ValueError):
        green_reopening_pull = 0.0
    live_engagement_state = dict(packet.get("live_engagement_state") or {})
    live_engagement_name = str(live_engagement_state.get("state") or "").strip()
    live_primary_move = str(live_engagement_state.get("primary_move") or "").strip()
    try:
        live_engagement_score = float(live_engagement_state.get("score") or 0.0)
    except (TypeError, ValueError):
        live_engagement_score = 0.0
    lightness_budget_state = dict(packet.get("lightness_budget_state") or {})
    lightness_budget_name = str(lightness_budget_state.get("state") or "").strip()
    try:
        lightness_banter_room = float(lightness_budget_state.get("banter_room") or 0.0)
    except (TypeError, ValueError):
        lightness_banter_room = 0.0
    try:
        lightness_playful_ceiling = float(lightness_budget_state.get("playful_ceiling") or 0.0)
    except (TypeError, ValueError):
        lightness_playful_ceiling = 0.0
    shared_moment_state = dict(packet.get("shared_moment_state") or {})
    shared_moment_name = str(shared_moment_state.get("state") or "").strip()
    shared_moment_kind = str(shared_moment_state.get("moment_kind") or "").strip()
    try:
        shared_moment_score = float(shared_moment_state.get("score") or 0.0)
    except (TypeError, ValueError):
        shared_moment_score = 0.0
    try:
        shared_moment_jointness = float(shared_moment_state.get("jointness") or 0.0)
    except (TypeError, ValueError):
        shared_moment_jointness = 0.0
    try:
        shared_moment_afterglow = float(shared_moment_state.get("afterglow") or 0.0)
    except (TypeError, ValueError):
        shared_moment_afterglow = 0.0
    shared_moment_room = max(
        0.0,
        min(
            1.0,
            shared_moment_score * 0.56
            + shared_moment_jointness * 0.24
            + shared_moment_afterglow * 0.2,
        ),
    )
    utterance_reason_packet = dict(packet.get("utterance_reason_packet") or {})
    utterance_reason_state = str(utterance_reason_packet.get("state") or "").strip()
    utterance_reason_offer = str(utterance_reason_packet.get("offer") or "").strip()
    utterance_reason_question_policy = str(
        utterance_reason_packet.get("question_policy") or ""
    ).strip()
    organism_state = dict(packet.get("organism_state") or {})
    organism_posture = str(organism_state.get("dominant_posture") or "").strip()
    try:
        organism_play_window = float(organism_state.get("play_window") or 0.0)
    except (TypeError, ValueError):
        organism_play_window = 0.0
    try:
        organism_expressive_readiness = float(
            organism_state.get("expressive_readiness") or 0.0
        )
    except (TypeError, ValueError):
        organism_expressive_readiness = 0.0
    surface_profile = dict(packet.get("surface_profile") or {})
    voice_texture = str(
        surface_profile.get("voice_texture")
        or packet.get("voice_texture")
        or ""
    ).strip()
    bright_style_open = (
        voice_texture in {"light_playful", "warm_companion"}
        or lightness_budget_name in {"open_play", "warm_only", "light_ok"}
    )
    bright_reason_open = (
        utterance_reason_state == "active"
        and utterance_reason_offer in {"brief_shared_smile", "small_shared_relief", "tiny_shared_win"}
        and utterance_reason_question_policy in {"", "none"}
    )
    bright_organism_open = (
        organism_posture in {"play", "open", "attune"}
        and organism_play_window >= 0.22
        and organism_expressive_readiness >= 0.28
    )
    explicit_bright_continuity = (
        live_engagement_name in {"pickup_comment", "riff_with_comment", "seed_topic"}
        and (
            lightness_budget_name in {"open_play", "warm_only", "light_ok"}
            or lightness_banter_room >= 0.22
            or lightness_playful_ceiling >= 0.22
            or voice_texture in {"light_playful", "warm_companion"}
            or constraints.allow_small_next_step
        )
        and (
            shared_moment_room >= 0.22
            or bright_reason_open
            or bright_organism_open
        )
    )
    bright_reentry_override = (
        recent_dialogue_kind == "reopening_thread"
        and explicit_bright_continuity
        and (
            shared_moment_kind in {"laugh", "relief", "pleasant_surprise", "tiny_win"}
            or utterance_reason_offer in {
                "brief_shared_smile",
                "small_shared_relief",
                "tiny_shared_win",
            }
        )
        and (
            shared_moment_room >= 0.3
            or recent_thread_carry >= 0.36
        )
    )
    anchor_hint = select_anchor_hint(
        (
            dominant_discussion_anchor,
            recent_dialogue_anchor,
            autobiographical_thread_anchor,
            issue_anchor,
            discussion_thread_anchor,
            autobiographical_thread_focus
            if autobiographical_thread_strength >= 0.46
            else "",
        ),
        limit=48,
    )

    if bright_reentry_override:
        preferred_act = (
            "light_bounce"
            if live_primary_move in {"pick_up_comment", "answer_visible_comment", "riff_current_comment"}
            or live_engagement_score >= 0.42
            or shared_moment_kind == "laugh"
            or utterance_reason_offer == "brief_shared_smile"
            else "shared_delight"
        )
        return TurnDelta(
            kind="bright_continuity",
            preferred_act=preferred_act,
            anchor_hint=anchor_hint,
            priority=0.64,
            rationale=tuple(
                item
                for item in (
                    f"recent_dialogue:{recent_dialogue_kind}",
                    "recent_dialogue:bright_reentry_override",
                    f"shared_moment:{shared_moment_kind}" if shared_moment_kind else "",
                    f"utterance_reason:{utterance_reason_offer}" if utterance_reason_offer else "",
                    f"live:{live_engagement_name}" if live_engagement_name else "",
                    f"lightness:{lightness_budget_name}" if lightness_budget_name else "",
                    f"organism:{organism_posture}" if organism_posture else "",
                )
                if item
            ),
        )

    if issue_kind == "pausing_issue" and issue_pause_readiness >= 0.38:
        preferred_act = (
            "leave_return_point_from_anchor"
            if anchor_hint and constraints.prefer_return_point and not constraints.allow_small_next_step
            else "leave_return_point"
            if constraints.prefer_return_point and not constraints.allow_small_next_step
            else "protect_talking_room"
        )
        return TurnDelta(
            kind="issue_pause",
            preferred_act=preferred_act,
            anchor_hint=anchor_hint,
            priority=0.58,
            rationale=tuple(
                item
                for item in (
                    f"issue:{issue_kind}",
                    "issue:pause_ready",
                )
                if item
            ),
        )
    if issue_kind == "exploring_issue" and issue_question_pressure >= 0.36:
        return TurnDelta(
            kind="issue_exploration",
            preferred_act="stay_with_present_need",
            priority=0.55,
            rationale=tuple(
                item
                for item in (
                    f"issue:{issue_kind}",
                    "issue:question_pressure",
                )
                if item
            ),
        )
    if discussion_thread_kind == "revisit_issue" and discussion_revisit_readiness >= 0.4:
        preferred_act = (
            "reopen_from_anchor"
            if anchor_hint
            else "leave_return_point"
            if constraints.prefer_return_point and not constraints.allow_small_next_step
            else "keep_shared_thread_visible"
        )
        return TurnDelta(
            kind="discussion_revisit",
            preferred_act=preferred_act,
            anchor_hint=anchor_hint,
            priority=0.64,
            rationale=tuple(
                item
                for item in (
                    f"discussion:{discussion_thread_kind}",
                    f"agenda_window:{agenda_window_state}" if agenda_window_state else "",
                    "discussion:revisit_ready",
                )
                if item
            ),
        )
    if discussion_thread_kind in {"active_issue", "fresh_issue"} and discussion_unresolved_pressure >= 0.28:
        return TurnDelta(
            kind="discussion_unresolved",
            preferred_act="stay_with_present_need",
            priority=0.57,
            rationale=tuple(
                item
                for item in (
                    f"discussion:{discussion_thread_kind}",
                    "discussion:unresolved",
                )
                if item
            ),
        )
    if recent_dialogue_kind == "reopening_thread" and recent_thread_carry >= 0.45:
        preferred_act = (
            "reopen_from_anchor"
            if anchor_hint
            else "leave_return_point"
            if constraints.prefer_return_point and not constraints.allow_small_next_step
            else "keep_shared_thread_visible"
        )
        return TurnDelta(
            kind="reopening_thread",
            preferred_act=preferred_act,
            anchor_hint=anchor_hint,
            priority=0.66,
            rationale=tuple(
                item
                for item in (
                    f"recent_dialogue:{recent_dialogue_kind}",
                    f"agenda_window:{agenda_window_state}" if agenda_window_state else "",
                    f"temporal:{temporal_membrane_mode}" if temporal_membrane_mode else "",
                    "recent_dialogue:reopen_pressure"
                    if recent_reopen_pressure >= 0.2
                    else "",
                )
                if item
            ),
        )
    if recent_dialogue_kind == "continuing_thread" and recent_thread_carry >= 0.34:
        if (
            (
                green_guardedness < 0.34
                and (
                    lightness_budget_name in {"open_play", "warm_only", "light_ok"}
                    or lightness_banter_room >= 0.22
                    or live_engagement_name in {"pickup_comment", "riff_with_comment", "seed_topic"}
                    or voice_texture in {"light_playful", "warm_companion"}
                    or constraints.allow_small_next_step
                )
            )
            or explicit_bright_continuity
        ):
            preferred_act = (
                "light_bounce"
                if live_primary_move in {"pick_up_comment", "answer_visible_comment", "riff_current_comment"}
                or live_engagement_score >= 0.42
                or shared_moment_kind == "laugh"
                or utterance_reason_offer == "brief_shared_smile"
                else "shared_delight"
            )
            return TurnDelta(
                kind="bright_continuity",
                preferred_act=preferred_act,
                priority=0.61,
                rationale=tuple(
                    item
                    for item in (
                        f"recent_dialogue:{recent_dialogue_kind}",
                        f"lightness:{lightness_budget_name}" if lightness_budget_name else "",
                        f"live:{live_engagement_name}" if live_engagement_name else "",
                        f"voice:{voice_texture}" if voice_texture in {"light_playful", "warm_companion"} else "",
                        "continuity:light_banter_room" if lightness_banter_room >= 0.22 else "",
                        "continuity:playful_ceiling" if lightness_playful_ceiling >= 0.22 else "",
                        f"shared_moment:{shared_moment_kind}" if shared_moment_room >= 0.22 and shared_moment_kind else "",
                        f"utterance_reason:{utterance_reason_offer}" if bright_reason_open else "",
                        f"organism:{organism_posture}" if bright_organism_open else "",
                    )
                    if item
                ),
            )
        return TurnDelta(
            kind="continuity_thread",
            preferred_act="keep_shared_thread_visible",
            priority=0.6,
            rationale=tuple(
                item
                for item in (
                    f"recent_dialogue:{recent_dialogue_kind}",
                    f"agenda_window:{agenda_window_state}" if agenda_window_state else "",
                )
                if item
            ),
        )
    if (
        green_affective_charge >= 0.44
        and green_guardedness >= 0.34
        and green_reopening_pull < 0.54
        and issue_question_pressure < 0.42
    ):
        return TurnDelta(
            kind="green_reflection_hold",
            preferred_act="stay_with_present_need",
            anchor_hint=anchor_hint,
            priority=0.58,
            rationale=tuple(
                item
                for item in (
                    "green:affective_charge",
                    "green:guarded_reflection",
                    f"issue:{issue_kind}" if issue_kind else "",
                )
                if item
            ),
        )
    if (
        autobiographical_thread_mode in {"unfinished_thread", "relational_lingering_thread"}
        and autobiographical_thread_strength >= 0.42
    ):
        preferred_act = (
            "reopen_from_anchor"
            if anchor_hint
            else "leave_return_point"
            if constraints.prefer_return_point and not constraints.allow_small_next_step
            else "keep_shared_thread_visible"
        )
        return TurnDelta(
            kind="autobiographical_reopen",
            preferred_act=preferred_act,
            anchor_hint=anchor_hint,
            priority=0.59,
            rationale=tuple(
                item
                for item in (
                    f"autobiographical:{autobiographical_thread_mode}",
                    f"agenda_window:{agenda_window_state}" if agenda_window_state else "",
                    "autobiographical:strong_thread",
                )
                if item
            ),
        )
    if (
        autobiographical_thread_mode in {"lingering_thread", "residual_lingering"}
        and autobiographical_thread_strength >= 0.5
        and constraints.prefer_return_point
        and not constraints.allow_small_next_step
    ):
        preferred_act = "leave_return_point_from_anchor" if anchor_hint else "leave_return_point"
        return TurnDelta(
            kind="autobiographical_return",
            preferred_act=preferred_act,
            anchor_hint=anchor_hint,
            priority=0.55,
            rationale=tuple(
                item
                for item in (
                    f"autobiographical:{autobiographical_thread_mode}",
                    f"agenda_window:{agenda_window_state}" if agenda_window_state else "",
                    "autobiographical:return_point",
                )
                if item
            ),
        )
    if (
        constraints.keep_thread_visible
        and constraints.allow_small_next_step
        and (green_guardedness < 0.34 or bright_style_open or explicit_bright_continuity)
    ):
        preferred_act = (
            "light_bounce"
            if live_primary_move in {"pick_up_comment", "answer_visible_comment", "riff_current_comment"}
            or live_engagement_score >= 0.42
            or shared_moment_kind == "laugh"
            or utterance_reason_offer == "brief_shared_smile"
            else "shared_delight"
        )
        return TurnDelta(
            kind="bright_continuity",
            preferred_act=preferred_act,
            priority=0.72,
            rationale=tuple(
                item
                for item in (
                    f"identity:{identity_arc_kind}" if identity_arc_kind else "",
                    "constraint:keep_thread_visible",
                    "constraint:allow_small_next_step",
                    f"voice:{voice_texture}" if voice_texture in {"light_playful", "warm_companion"} else "",
                    f"lightness:{lightness_budget_name}" if lightness_budget_name in {"open_play", "warm_only", "light_ok"} else "",
                    f"shared_moment:{shared_moment_kind}" if shared_moment_room >= 0.22 and shared_moment_kind else "",
                    f"utterance_reason:{utterance_reason_offer}" if bright_reason_open else "",
                    f"organism:{organism_posture}" if bright_organism_open else "",
                )
                if item
            ),
        )
    if constraints.keep_thread_visible:
        return TurnDelta(
            kind="continuity_thread",
            preferred_act="keep_shared_thread_visible",
            priority=0.78,
            rationale=tuple(
                item
                for item in (
                    f"identity:{identity_arc_kind}" if identity_arc_kind else "",
                    "constraint:keep_thread_visible",
                )
                if item
            ),
        )
    if constraints.prefer_return_point and not constraints.allow_small_next_step:
        return TurnDelta(
            kind="return_point",
            preferred_act="leave_return_point",
            priority=0.68,
            rationale=tuple(
                item
                for item in (
                    f"agenda_window:{agenda_window_state}" if agenda_window_state else "",
                    f"temporal:{temporal_membrane_mode}" if temporal_membrane_mode else "",
                    "constraint:prefer_return_point",
                )
                if item
            ),
        )
    if constraints.allow_small_next_step:
        return TurnDelta(
            kind="small_step",
            preferred_act="pace_match",
            priority=0.56,
            rationale=tuple(
                item
                for item in (
                    f"strategy:{response_strategy}" if response_strategy else "",
                    "constraint:allow_small_next_step",
                )
                if item
            ),
        )
    if constraints.avoid_obvious_advice:
        return TurnDelta(
            kind="stay_present_need",
            preferred_act="stay_with_present_need",
            priority=0.52,
            rationale=("constraint:avoid_obvious_advice",),
        )
    if constraints.avoid_overclosure:
        return TurnDelta(
            kind="boundary_soft_close",
            preferred_act="protect_talking_room",
            priority=0.44,
            rationale=("constraint:avoid_overclosure",),
        )
    return TurnDelta()


def coerce_turn_delta(payload: Mapping[str, Any] | None) -> TurnDelta:
    source = dict(payload or {})
    rationale = tuple(
        str(item).strip()
        for item in source.get("rationale") or []
        if str(item).strip()
    )
    try:
        priority = float(source.get("priority") or 0.0)
    except (TypeError, ValueError):
        priority = 0.0
    return TurnDelta(
        kind=str(source.get("kind") or "").strip(),
        preferred_act=str(source.get("preferred_act") or "").strip(),
        anchor_hint=str(source.get("anchor_hint") or "").strip(),
        priority=priority,
        rationale=rationale,
    )


def _nested_state(payload: Any) -> str:
    if isinstance(payload, Mapping):
        return str(payload.get("state") or "").strip()
    return str(payload or "").strip()
