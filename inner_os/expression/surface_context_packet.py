from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_list(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    cleaned: list[str] = []
    for value in values:
        text = _clean_text(value)
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _bool_value(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return bool(value)


@dataclass(frozen=True)
class SurfaceContextPacket:
    conversation_phase: str = ""
    shared_core: dict[str, Any] = field(default_factory=dict)
    response_role: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    surface_profile: dict[str, Any] = field(default_factory=dict)
    source_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_phase": self.conversation_phase,
            "shared_core": dict(self.shared_core),
            "response_role": dict(self.response_role),
            "constraints": dict(self.constraints),
            "surface_profile": dict(self.surface_profile),
            "source_state": dict(self.source_state),
        }


def build_surface_context_packet(
    *,
    recent_dialogue_state: Mapping[str, Any] | None = None,
    discussion_thread_state: Mapping[str, Any] | None = None,
    issue_state: Mapping[str, Any] | None = None,
    turn_delta: Mapping[str, Any] | None = None,
    interaction_constraints: Mapping[str, Any] | None = None,
    boundary_transform: Mapping[str, Any] | None = None,
    residual_reflection: Mapping[str, Any] | None = None,
    surface_profile: Mapping[str, Any] | None = None,
    contact_reflection_state: Mapping[str, Any] | None = None,
    green_kernel_composition: Mapping[str, Any] | None = None,
    dialogue_context: Mapping[str, Any] | None = None,
) -> SurfaceContextPacket:
    recent = dict(recent_dialogue_state or {})
    discussion = dict(discussion_thread_state or {})
    issue = dict(issue_state or {})
    delta = dict(turn_delta or {})
    constraints = dict(interaction_constraints or {})
    boundary = dict(boundary_transform or {})
    residual = dict(residual_reflection or {})
    profile = dict(surface_profile or {})
    contact_reflection = dict(contact_reflection_state or {})
    green = dict((green_kernel_composition or {}).get("field") or {})
    dialogue = dict(dialogue_context or {})

    conversation_phase = (
        _clean_text(delta.get("kind"))
        or _clean_text(recent.get("state"))
        or _clean_text(discussion.get("state"))
        or _clean_text(issue.get("state"))
    )
    anchor = (
        _clean_text(delta.get("anchor_hint"))
        or _clean_text(discussion.get("topic_anchor"))
        or _clean_text(recent.get("recent_anchor"))
        or _clean_text(issue.get("issue_anchor"))
    )
    shared_core = {
        "anchor": anchor,
        "already_shared": _clean_list(
            [
                anchor,
                _clean_text(dialogue.get("user_text")),
            ]
        ),
        "not_yet_shared": _clean_list(
            [
                _clean_text(residual.get("focus")),
                *_clean_list(residual.get("reasons")),
            ]
        ),
    }
    primary_act = _clean_text(delta.get("preferred_act"))
    reflection_style = _clean_text(contact_reflection.get("reflection_style"))
    response_role = {
        "primary": primary_act,
        "secondary": reflection_style,
    }
    max_questions = 0
    if primary_act.startswith("gentle_question") or reflection_style == "reflect_then_question":
        max_questions = 1
    constraints_payload = {
        "no_generic_clarification": True,
        "no_advice": _bool_value(constraints.get("avoid_obvious_advice"), default=True),
        "max_questions": max_questions,
        "keep_thread_visible": _bool_value(constraints.get("keep_thread_visible"), default=False),
        "prefer_return_point": _bool_value(constraints.get("prefer_return_point"), default=False),
        "boundary_style": _clean_text(boundary.get("surface_mode")),
    }
    surface_payload = {
        "response_length": _clean_text(profile.get("response_length")),
        "cultural_register": _clean_text(profile.get("cultural_register")),
        "group_register": _clean_text(profile.get("group_register")),
        "sentence_temperature": _clean_text(profile.get("sentence_temperature")),
        "surface_mode": _clean_text(profile.get("surface_mode")),
    }
    source_state = {
        "recent_dialogue_state": _clean_text(recent.get("state")),
        "discussion_thread_state": _clean_text(discussion.get("state")),
        "issue_state": _clean_text(issue.get("state")),
        "turn_delta_kind": _clean_text(delta.get("kind")),
        "green_guardedness": float(green.get("guardedness") or 0.0),
        "green_reopening_pull": float(green.get("reopening_pull") or 0.0),
        "green_affective_charge": float(green.get("affective_charge") or 0.0),
    }
    return SurfaceContextPacket(
        conversation_phase=conversation_phase,
        shared_core=shared_core,
        response_role=response_role,
        constraints=constraints_payload,
        surface_profile=surface_payload,
        source_state=source_state,
    )
