from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _float01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    if numeric <= 0.0:
        return 0.0
    if numeric >= 1.0:
        return 1.0
    return float(numeric)


@dataclass(frozen=True)
class GroupRelationArcSummary:
    summary_version: str = "v1"
    arc_kind: str = ""
    phase: str = "forming"
    summary: str = ""
    group_thread_id: str = ""
    topology_focus: str = ""
    boundary_mode: str = ""
    reentry_window_focus: str = ""
    dominant_person_id: str = ""
    social_role: str = ""
    community_id: str = ""
    culture_id: str = ""
    learning_mode_focus: str = ""
    social_experiment_focus: str = ""
    open_tension: str = ""
    stability: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_version": self.summary_version,
            "arc_kind": self.arc_kind,
            "phase": self.phase,
            "summary": self.summary,
            "group_thread_id": self.group_thread_id,
            "topology_focus": self.topology_focus,
            "boundary_mode": self.boundary_mode,
            "reentry_window_focus": self.reentry_window_focus,
            "dominant_person_id": self.dominant_person_id,
            "social_role": self.social_role,
            "community_id": self.community_id,
            "culture_id": self.culture_id,
            "learning_mode_focus": self.learning_mode_focus,
            "social_experiment_focus": self.social_experiment_focus,
            "open_tension": self.open_tension,
            "stability": round(float(self.stability), 4),
        }


class GroupRelationArcSummaryBuilder:
    def build(self, report: Mapping[str, Any] | None) -> GroupRelationArcSummary:
        payload = dict(report or {})
        relation_arc = dict(payload.get("inner_os_relation_arc_summary") or {})
        relation_arc_registry = dict(payload.get("inner_os_relation_arc_registry_summary") or {})
        group_thread_registry = dict(payload.get("inner_os_group_thread_registry_summary") or {})
        partner_relation = dict(payload.get("inner_os_partner_relation_summary") or {})

        source = self._source_relation_arc(
            relation_arc=relation_arc,
            relation_arc_registry=relation_arc_registry,
        )
        group_thread_id = _text(source.get("group_thread_id")) or _text(group_thread_registry.get("dominant_thread_id"))
        if not group_thread_id:
            return GroupRelationArcSummary()

        topology_focus = (
            _text(source.get("topology_focus"))
            or _text(payload.get("inner_os_sleep_group_thread_focus"))
            or group_thread_id.split(":", 1)[0]
        )
        reentry_window_focus = _text(payload.get("inner_os_sleep_agenda_window_focus"))
        open_tension = _text(source.get("open_tension"))
        boundary_mode = self._boundary_mode(
            topology_focus=topology_focus,
            reentry_window_focus=reentry_window_focus,
            open_tension=open_tension,
        )
        arc_kind = _text(source.get("arc_kind")) or "group_thread_continuity"
        phase = _text(source.get("phase")) or ("holding" if group_thread_id else "forming")
        stability = _float01(
            source.get("stability")
            or (group_thread_registry.get("thread_scores") or {}).get(group_thread_id)
        )
        dominant_person_id = _text(source.get("related_person_id")) or _text(partner_relation.get("person_id"))
        social_role = _text(source.get("social_role")) or _text(partner_relation.get("social_role"))
        community_id = _text(source.get("community_id")) or _text(partner_relation.get("community_id"))
        culture_id = _text(source.get("culture_id")) or _text(partner_relation.get("culture_id"))
        learning_mode_focus = _text(source.get("learning_mode_focus")) or _text(payload.get("inner_os_sleep_learning_mode_focus"))
        social_experiment_focus = _text(source.get("social_experiment_focus")) or _text(payload.get("inner_os_sleep_social_experiment_focus"))
        summary = _text(source.get("summary")) or self._summary(
            arc_kind=arc_kind,
            phase=phase,
            group_thread_id=group_thread_id,
            boundary_mode=boundary_mode,
            dominant_person_id=dominant_person_id,
        )

        return GroupRelationArcSummary(
            arc_kind=arc_kind,
            phase=phase,
            summary=summary,
            group_thread_id=group_thread_id,
            topology_focus=topology_focus,
            boundary_mode=boundary_mode,
            reentry_window_focus=reentry_window_focus,
            dominant_person_id=dominant_person_id,
            social_role=social_role,
            community_id=community_id,
            culture_id=culture_id,
            learning_mode_focus=learning_mode_focus,
            social_experiment_focus=social_experiment_focus,
            open_tension=open_tension,
            stability=stability,
        )

    def _source_relation_arc(
        self,
        *,
        relation_arc: Mapping[str, Any],
        relation_arc_registry: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if _text(relation_arc.get("group_thread_id")):
            return relation_arc
        for item in relation_arc_registry.get("top_arcs") or []:
            if isinstance(item, Mapping) and _text(item.get("group_thread_id")):
                return item
        return {}

    def _boundary_mode(
        self,
        *,
        topology_focus: str,
        reentry_window_focus: str,
        open_tension: str,
    ) -> str:
        if reentry_window_focus == "next_private_window":
            return "public_to_private_hold"
        if reentry_window_focus == "next_same_group_window":
            return "same_group_reentry"
        if reentry_window_focus == "next_same_culture_window":
            return "same_culture_reentry"
        if open_tension == "public_boundary" or topology_focus in {"public_visible", "hierarchical"}:
            return "public_boundary"
        if topology_focus == "threaded_group":
            return "thread_continuity"
        return ""

    def _summary(
        self,
        *,
        arc_kind: str,
        phase: str,
        group_thread_id: str,
        boundary_mode: str,
        dominant_person_id: str,
    ) -> str:
        base = {
            "group_thread_continuity": "the same group thread is being kept available for a later return",
            "repairing_relation": "repair is moving through a shared group thread in small steps",
            "deferred_reentry": "a group-thread topic is being held for a better reentry window",
            "trusted_relation": "a trusted group thread is stabilizing",
            "emerging_relation": "a group thread is beginning to take shape",
        }.get(arc_kind, "a shared group thread is staying active")
        parts = [base, f"phase={phase}", f"group={group_thread_id}"]
        if boundary_mode:
            parts.append(f"boundary={boundary_mode}")
        if dominant_person_id:
            parts.append(f"person={dominant_person_id}")
        return " / ".join(parts)
