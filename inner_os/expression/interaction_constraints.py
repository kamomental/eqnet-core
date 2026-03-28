from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class InteractionConstraints:
    avoid_obvious_advice: bool = False
    avoid_overclosure: bool = False
    prefer_return_point: bool = False
    keep_thread_visible: bool = False
    allow_small_next_step: bool = False
    prefer_acknowledge_before_extension: bool = False
    dominant_reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "avoid_obvious_advice": self.avoid_obvious_advice,
            "avoid_overclosure": self.avoid_overclosure,
            "prefer_return_point": self.prefer_return_point,
            "keep_thread_visible": self.keep_thread_visible,
            "allow_small_next_step": self.allow_small_next_step,
            "prefer_acknowledge_before_extension": self.prefer_acknowledge_before_extension,
            "dominant_reasons": list(self.dominant_reasons),
        }


def derive_interaction_constraints(
    interaction_policy: Mapping[str, Any] | None,
) -> InteractionConstraints:
    packet = dict(interaction_policy or {})
    grice_state = _nested_state(packet.get("grice_guard_state"))
    agenda_window_state = _nested_state(packet.get("agenda_window_state"))
    relational_continuity_state = _nested_state(packet.get("relational_continuity_state"))
    learning_mode_state = _nested_state(packet.get("learning_mode_state"))
    social_topology_state = _nested_state(packet.get("social_topology_state"))
    response_strategy = str(packet.get("response_strategy") or "").strip()
    primary_operation_kind = str(
        (packet.get("primary_object_operation") or {}).get("operation_kind") or ""
    ).strip()
    identity_arc_kind = str(packet.get("identity_arc_kind") or "").strip()
    identity_arc_open_tension = str(packet.get("identity_arc_open_tension") or "").strip()
    temporal_membrane_mode = str(
        packet.get("temporal_membrane_mode")
        or packet.get("temporal_membrane_focus")
        or ""
    ).strip()

    avoid_obvious_advice = grice_state in {
        "hold_obvious_advice",
        "attune_without_repeating",
    }
    keep_thread_visible = (
        identity_arc_kind in {"repairing_bond", "holding_thread", "shared_place_thread"}
        or relational_continuity_state in {"holding_thread", "reopening", "co_regulating"}
    )
    prefer_return_point = (
        agenda_window_state in {
            "next_private_window",
            "next_same_group_window",
            "next_same_culture_window",
            "opportunistic_reentry",
            "long_hold",
        }
        or temporal_membrane_mode in {
            "reentry",
            "coherent_reentry",
            "same_group_reentry",
            "same_culture_reentry",
        }
        or identity_arc_open_tension in {
            "timing_sensitive_reentry",
            "guarded_closeness",
            "careful_repair",
        }
    )
    allow_small_next_step = (
        response_strategy == "shared_world_next_step"
        or primary_operation_kind == "offer_small_next_step"
    )
    prefer_acknowledge_before_extension = response_strategy in {
        "attune_then_extend",
        "repair_then_attune",
        "reflect_without_settling",
    } or keep_thread_visible
    avoid_overclosure = (
        avoid_obvious_advice
        or prefer_return_point
        or learning_mode_state in {"observe_only", "hold_and_wait"}
        or social_topology_state in {"public_visible", "hierarchical"}
    )

    reasons: list[str] = []
    if avoid_obvious_advice:
        reasons.append(f"grice:{grice_state}")
    if keep_thread_visible:
        reasons.append("continuity:thread")
    if prefer_return_point:
        reasons.append("timing:return_point")
    if allow_small_next_step:
        reasons.append("strategy:small_step")
    if avoid_overclosure:
        reasons.append("boundary:soft_close")
    if prefer_acknowledge_before_extension:
        reasons.append("opening:acknowledge_first")

    return InteractionConstraints(
        avoid_obvious_advice=avoid_obvious_advice,
        avoid_overclosure=avoid_overclosure,
        prefer_return_point=prefer_return_point,
        keep_thread_visible=keep_thread_visible,
        allow_small_next_step=allow_small_next_step,
        prefer_acknowledge_before_extension=prefer_acknowledge_before_extension,
        dominant_reasons=tuple(reasons),
    )


def coerce_interaction_constraints(
    payload: Mapping[str, Any] | None,
) -> InteractionConstraints:
    source = dict(payload or {})
    reasons = tuple(
        str(item).strip()
        for item in source.get("dominant_reasons") or []
        if str(item).strip()
    )
    return InteractionConstraints(
        avoid_obvious_advice=bool(source.get("avoid_obvious_advice", False)),
        avoid_overclosure=bool(source.get("avoid_overclosure", False)),
        prefer_return_point=bool(source.get("prefer_return_point", False)),
        keep_thread_visible=bool(source.get("keep_thread_visible", False)),
        allow_small_next_step=bool(source.get("allow_small_next_step", False)),
        prefer_acknowledge_before_extension=bool(
            source.get("prefer_acknowledge_before_extension", False)
        ),
        dominant_reasons=reasons,
    )


def _nested_state(payload: Any) -> str:
    if isinstance(payload, Mapping):
        return str(payload.get("state") or "").strip()
    return str(payload or "").strip()
