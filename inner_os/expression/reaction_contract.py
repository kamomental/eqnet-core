from __future__ import annotations

from collections.abc import Iterator, Mapping as MappingABC
from dataclasses import dataclass, field
from typing import Any, Mapping


_REACTION_CONTRACT_CORE_KEYS = frozenset(
    {
        "stance",
        "scale",
        "initiative",
        "question_budget",
        "interpretation_budget",
        "response_channel",
        "timing_mode",
        "continuity_mode",
        "distance_mode",
        "closure_mode",
        "reason_tags",
    }
)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _float01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def _as_text_list(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    ordered: list[str] = []
    for value in values:
        text = _clean_text(value)
        if text and text not in ordered:
            ordered.append(text)
    return ordered


def _state_name(value: Any) -> str:
    if isinstance(value, Mapping):
        return _clean_text(value.get("state"))
    return _clean_text(value)


def _dedupe(values: list[str]) -> list[str]:
    ordered: list[str] = []
    for value in values:
        text = _clean_text(value)
        if text and text not in ordered:
            ordered.append(text)
    return ordered


def _closure_reason_tags(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    tags: list[str] = []
    for key in (
        "generated_constraints",
        "generated_affordances",
        "inhibition_reasons",
        "uncertainty_reasons",
    ):
        raw_values = value.get(key)
        if not isinstance(raw_values, (list, tuple)):
            continue
        for raw_value in raw_values:
            text = _clean_text(raw_value)
            if text:
                tags.append(f"closure:{text}")
    return _dedupe(tags)


@dataclass(frozen=True)
class ReactionContract(MappingABC[str, object]):
    stance: str = ""
    scale: str = ""
    initiative: str = ""
    question_budget: int = 0
    interpretation_budget: str = ""
    response_channel: str = ""
    timing_mode: str = ""
    continuity_mode: str = ""
    distance_mode: str = ""
    closure_mode: str = ""
    reason_tags: list[str] = field(default_factory=list)
    extras: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "stance": self.stance,
            "scale": self.scale,
            "initiative": self.initiative,
            "question_budget": self.question_budget,
            "interpretation_budget": self.interpretation_budget,
            "response_channel": self.response_channel,
            "timing_mode": self.timing_mode,
            "continuity_mode": self.continuity_mode,
            "distance_mode": self.distance_mode,
            "closure_mode": self.closure_mode,
            "reason_tags": list(self.reason_tags),
        }
        data.update(self.extras)
        return data

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def coerce_reaction_contract(
    value: Mapping[str, Any] | ReactionContract | None,
) -> ReactionContract:
    if isinstance(value, ReactionContract):
        return value
    packet = dict(value or {})
    extras = {key: packet[key] for key in packet.keys() - _REACTION_CONTRACT_CORE_KEYS}
    return ReactionContract(
        stance=_clean_text(packet.get("stance")),
        scale=_clean_text(packet.get("scale")),
        initiative=_clean_text(packet.get("initiative")),
        question_budget=int(packet.get("question_budget") or 0),
        interpretation_budget=_clean_text(packet.get("interpretation_budget")),
        response_channel=_clean_text(packet.get("response_channel")),
        timing_mode=_clean_text(packet.get("timing_mode")),
        continuity_mode=_clean_text(packet.get("continuity_mode")),
        distance_mode=_clean_text(packet.get("distance_mode")),
        closure_mode=_clean_text(packet.get("closure_mode")),
        reason_tags=_as_text_list(packet.get("reason_tags")),
        extras=extras,
    )


def derive_reaction_contract(
    *,
    interaction_policy: Mapping[str, Any] | None = None,
    action_posture: Mapping[str, Any] | None = None,
    actuation_plan: Mapping[str, Any] | None = None,
    discourse_shape: Mapping[str, Any] | None = None,
    surface_context_packet: Mapping[str, Any] | None = None,
    turn_delta: Mapping[str, Any] | None = None,
) -> ReactionContract:
    policy = dict(interaction_policy or {})
    posture = dict(action_posture or {})
    actuation = dict(actuation_plan or {})
    shape = dict(discourse_shape or {})
    packet = dict(surface_context_packet or {})
    delta = dict(turn_delta or {})

    response_action = dict(dict(policy.get("conversation_contract") or {}).get("response_action_now") or {})
    constraints = dict(packet.get("constraints") or {})
    source_state = dict(packet.get("source_state") or {})
    packet_surface_profile = dict(packet.get("surface_profile") or {})
    closure_packet = dict(packet.get("closure_packet") or {})

    strategy = _clean_text(policy.get("response_strategy"))
    execution_mode = _clean_text(actuation.get("execution_mode"))
    response_channel = _clean_text(actuation.get("response_channel"))
    wait_before_action = _clean_text(actuation.get("wait_before_action"))
    boundary_mode = _clean_text(posture.get("boundary_mode"))
    shape_id = _clean_text(
        shape.get("shape_id")
        or source_state.get("discourse_shape_id")
        or packet_surface_profile.get("discourse_shape_id")
    )
    conversation_phase = _clean_text(packet.get("conversation_phase"))
    recent_dialogue_state = _state_name(
        source_state.get("recent_dialogue_state") or policy.get("recent_dialogue_state")
    )
    preserve = _clean_text(
        source_state.get("utterance_reason_preserve")
        or actuation.get("utterance_reason_preserve")
        or posture.get("utterance_reason_preserve")
    )
    offer = _clean_text(source_state.get("utterance_reason_offer"))
    response_length = _clean_text(
        packet_surface_profile.get("response_length")
        or dict(policy.get("surface_profile") or {}).get("response_length")
    )
    timing_bias = _clean_text(
        dict(actuation.get("nonverbal_response_state") or {}).get("timing_bias")
    )
    question_budget_source = shape.get("question_budget")
    if question_budget_source is None:
        question_budget_source = constraints.get("max_questions")
    if question_budget_source is None:
        question_budget_source = posture.get("question_budget")
    if question_budget_source is None:
        question_budget_source = response_action.get("question_budget")
    question_budget = int(question_budget_source or 0)

    protective_tension = _float01(source_state.get("organism_protective_tension"))
    common_ground = _float01(source_state.get("joint_common_ground"))
    social_pressure = _float01(source_state.get("external_field_social_pressure"))
    organism_social_mode = _clean_text(
        source_state.get("organism_social_mode")
        or source_state.get("external_field_social_mode")
    )
    social_topology = _clean_text(posture.get("social_topology_name"))
    shared_presence_mode = _clean_text(source_state.get("shared_presence_mode"))
    shared_presence_co_presence = _float01(source_state.get("shared_presence_co_presence"))
    has_shared_presence_boundary_stability = (
        "shared_presence_boundary_stability" in source_state
    )
    shared_presence_boundary_stability = _float01(
        source_state.get("shared_presence_boundary_stability")
    )
    self_other_dominant_attribution = _clean_text(
        source_state.get("self_other_dominant_attribution")
    )
    self_other_unknown_likelihood = _float01(
        source_state.get("self_other_unknown_likelihood")
    )
    subjective_scene_shared_scene_potential = _float01(
        source_state.get("subjective_scene_shared_scene_potential")
    )
    subjective_scene_familiarity = _float01(source_state.get("subjective_scene_familiarity"))
    subjective_scene_anchor_frame = _clean_text(source_state.get("subjective_scene_anchor_frame"))
    shared_inhabitation_signal = max(
        shared_presence_co_presence,
        subjective_scene_shared_scene_potential,
        common_ground,
    )
    guarded_self_view_signal = max(
        self_other_unknown_likelihood,
        max(0.0, 1.0 - shared_presence_boundary_stability)
        if has_shared_presence_boundary_stability
        else 0.0,
    )

    scale = "medium"
    if response_channel in {"hold", "backchannel"}:
        scale = "micro"
    elif (
        preserve == "keep_it_small"
        or offer == "brief_shared_smile"
        or shape_id == "bright_bounce"
        or response_length == "short"
    ):
        scale = "small"
    elif response_length == "forward_leaning":
        scale = "medium"

    interpretation_budget = "low"
    if (
        preserve == "keep_it_small"
        or offer == "brief_shared_smile"
        or shape_id == "bright_bounce"
        or response_channel in {"backchannel", "hold"}
    ):
        interpretation_budget = "none"
    elif boundary_mode in {"contain", "stabilize", "shield"} or strategy in {
        "respectful_wait",
        "repair_then_attune",
    }:
        interpretation_budget = "low"
    elif shape_id in {"reflect_step"}:
        interpretation_budget = "medium"
    if (
        shared_inhabitation_signal >= 0.58
        and self_other_dominant_attribution == "shared"
        and shared_presence_boundary_stability >= 0.42
        and interpretation_budget == "low"
    ):
        interpretation_budget = "none"
    if guarded_self_view_signal >= 0.58:
        interpretation_budget = "low"

    stance = "receive"
    if response_channel == "hold":
        stance = "hold"
    elif response_channel == "defer":
        stance = "wait"
    elif "repair" in strategy or "repair" in execution_mode:
        stance = "repair"
    elif strategy == "respectful_wait":
        stance = "witness"
    elif response_channel == "backchannel" or strategy == "shared_world_next_step" or execution_mode == "shared_progression":
        stance = "join"
    if (
        stance in {"receive", "witness"}
        and self_other_dominant_attribution == "shared"
        and shared_inhabitation_signal >= 0.6
        and guarded_self_view_signal < 0.55
    ):
        stance = "join"
    if stance in {"receive", "join"} and guarded_self_view_signal >= 0.64:
        stance = "hold"

    initiative = "receive"
    if response_channel in {"hold", "defer"}:
        initiative = "yield"
    elif execution_mode == "shared_progression" or strategy == "shared_world_next_step":
        initiative = "co_move"
    elif question_budget > 0:
        initiative = "guide"
    if stance == "join" and initiative == "receive":
        initiative = "co_move"
    if stance in {"hold", "wait"}:
        initiative = "yield"

    timing_mode = "immediate"
    if response_channel == "hold":
        timing_mode = "held_open"
    elif response_channel == "defer":
        timing_mode = "delayed"
    elif wait_before_action in {"brief", "short", "measured"}:
        timing_mode = "brief_wait"
    elif response_channel == "backchannel" or timing_bias in {"gentle_overlap", "quick_ack"}:
        timing_mode = "quick_ack"

    continuity_mode = "open"
    if conversation_phase in {"issue_pause", "reopening_thread", "thread_reopening"} or recent_dialogue_state in {
        "reopening_thread",
        "thread_reopening",
    }:
        continuity_mode = "reopen"
    elif conversation_phase in {"bright_continuity", "continuing_thread"} or recent_dialogue_state == "continuing_thread":
        continuity_mode = "continue"
    elif conversation_phase in {"fresh_opening", "clarify"} or recent_dialogue_state == "fresh_opening":
        continuity_mode = "fresh"
    if continuity_mode == "open" and subjective_scene_anchor_frame in {"shared_margin", "front_field"}:
        continuity_mode = "continue"

    distance_mode = "steady"
    if (
        boundary_mode in {"contain", "stabilize", "shield"}
        or response_channel in {"hold", "defer"}
        or protective_tension >= 0.45
    ):
        distance_mode = "guarded"
    elif social_topology in {"public_visible", "hierarchical"} or social_pressure >= 0.45:
        distance_mode = "respectful"
    elif organism_social_mode == "near" or common_ground >= 0.55:
        distance_mode = "near"
    if (
        distance_mode == "steady"
        and self_other_dominant_attribution == "shared"
        and shared_inhabitation_signal >= 0.56
        and subjective_scene_familiarity >= 0.42
    ):
        distance_mode = "near"
    if guarded_self_view_signal >= 0.62:
        distance_mode = "guarded"

    closure_mode = "soft_close"
    if response_channel in {"hold", "defer"} or strategy == "respectful_wait":
        closure_mode = "leave_open"
    elif bool(constraints.get("prefer_return_point")) or bool(constraints.get("keep_thread_visible")):
        closure_mode = "return_point_open"
    elif shape_id == "bright_bounce":
        closure_mode = "open_light"
    elif shape_id in {"reflect_hold", "anchor_reopen", "opening_support"}:
        closure_mode = "soft_open"

    reason_tags = _dedupe(
        [
            strategy,
            execution_mode,
            response_channel,
            wait_before_action,
            shape_id,
            conversation_phase,
            recent_dialogue_state,
            preserve,
            offer,
            _clean_text(source_state.get("utterance_reason_relation_frame")),
            _clean_text(source_state.get("utterance_reason_causal_frame")),
            _clean_text(source_state.get("utterance_reason_memory_frame")),
            *_closure_reason_tags(closure_packet),
            _clean_text(delta.get("kind")),
            _clean_text(delta.get("preferred_act")),
        ]
    )

    extras = {
        "shape_id": shape_id,
        "strategy": strategy,
        "execution_mode": execution_mode,
        "wait_before_action": wait_before_action,
        "shared_presence_mode": shared_presence_mode,
        "shared_presence_co_presence": shared_presence_co_presence,
        "shared_presence_boundary_stability": shared_presence_boundary_stability,
        "self_other_dominant_attribution": self_other_dominant_attribution,
        "self_other_unknown_likelihood": self_other_unknown_likelihood,
        "subjective_scene_shared_scene_potential": subjective_scene_shared_scene_potential,
        "subjective_scene_familiarity": subjective_scene_familiarity,
        "subjective_scene_anchor_frame": subjective_scene_anchor_frame,
        "closure_packet": closure_packet,
    }

    return ReactionContract(
        stance=stance,
        scale=scale,
        initiative=initiative,
        question_budget=question_budget,
        interpretation_budget=interpretation_budget,
        response_channel=response_channel,
        timing_mode=timing_mode,
        continuity_mode=continuity_mode,
        distance_mode=distance_mode,
        closure_mode=closure_mode,
        reason_tags=reason_tags,
        extras=extras,
    )
