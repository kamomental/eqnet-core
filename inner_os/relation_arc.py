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


def _driver(*parts: str) -> str:
    return ":".join(part for part in parts if part)


@dataclass(frozen=True)
class RelationArcSummary:
    summary_version: str = "v1"
    arc_kind: str = ""
    phase: str = "forming"
    summary: str = ""
    dominant_driver: str = ""
    supporting_drivers: tuple[str, ...] = ()
    open_tension: str = ""
    stability: float = 0.0
    related_person_id: str = ""
    group_thread_id: str = ""
    social_role: str = ""
    community_id: str = ""
    culture_id: str = ""
    topology_focus: str = ""
    learning_mode_focus: str = ""
    social_experiment_focus: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_version": self.summary_version,
            "arc_kind": self.arc_kind,
            "phase": self.phase,
            "summary": self.summary,
            "dominant_driver": self.dominant_driver,
            "supporting_drivers": list(self.supporting_drivers),
            "open_tension": self.open_tension,
            "stability": round(float(self.stability), 4),
            "related_person_id": self.related_person_id,
            "group_thread_id": self.group_thread_id,
            "social_role": self.social_role,
            "community_id": self.community_id,
            "culture_id": self.culture_id,
            "topology_focus": self.topology_focus,
            "learning_mode_focus": self.learning_mode_focus,
            "social_experiment_focus": self.social_experiment_focus,
        }


class RelationArcSummaryBuilder:
    def build(self, report: Mapping[str, Any] | None) -> RelationArcSummary:
        payload = dict(report or {})
        partner_summary = dict(payload.get("inner_os_partner_relation_summary") or {})
        partner_registry = dict(payload.get("inner_os_partner_relation_registry_summary") or {})
        group_thread_summary = dict(payload.get("inner_os_group_thread_registry_summary") or {})
        agenda_summary = dict(payload.get("inner_os_agenda_summary") or {})
        commitment_summary = dict(payload.get("inner_os_commitment_summary") or {})

        person_id = _text(partner_summary.get("person_id")) or _text(partner_registry.get("dominant_person_id"))
        group_thread_id = _text(group_thread_summary.get("dominant_thread_id"))
        social_role = _text(partner_summary.get("social_role"))
        social_interpretation = _text(partner_summary.get("social_interpretation"))
        community_id = _text(partner_summary.get("community_id"))
        culture_id = _text(partner_summary.get("culture_id"))
        attachment = _float01(partner_summary.get("attachment"))
        familiarity = _float01(partner_summary.get("familiarity"))
        trust_memory = _float01(partner_summary.get("trust_memory"))
        relation_strength = _float01(partner_summary.get("strength"))
        registry_total_people = int(partner_registry.get("total_people") or 0)
        group_total_threads = int(group_thread_summary.get("total_threads") or 0)
        learning_mode_focus = _text(payload.get("inner_os_same_turn_learning_mode_state")) or _text(payload.get("inner_os_sleep_learning_mode_focus"))
        social_experiment_focus = _text(payload.get("inner_os_same_turn_social_experiment_state")) or _text(payload.get("inner_os_sleep_social_experiment_focus"))
        agenda_state = _text(payload.get("inner_os_same_turn_agenda_state")) or _text(agenda_summary.get("dominant_agenda"))
        agenda_reason = _text(payload.get("inner_os_same_turn_agenda_reason")) or _text(agenda_summary.get("dominant_reason"))
        agenda_window_focus = _text(payload.get("inner_os_sleep_agenda_window_focus"))
        agenda_window_reason = _text(payload.get("inner_os_sleep_agenda_window_reason"))
        commitment_target = _text(commitment_summary.get("dominant_target"))
        commitment_state = _text(commitment_summary.get("dominant_state")) or "waver"
        topology_focus = _text(payload.get("inner_os_sleep_group_thread_focus"))
        if not topology_focus and group_thread_id:
            topology_focus = group_thread_id.split(":", 1)[0]
        if not topology_focus and registry_total_people > 1:
            topology_focus = "threaded_group"
        if not topology_focus and person_id:
            topology_focus = "one_to_one"

        supporting_drivers = tuple(
            item
            for item in (
                _driver("person", person_id),
                _driver("group", group_thread_id),
                _driver("role", social_role),
                _driver("agenda", agenda_state),
                _driver("learning", learning_mode_focus),
                _driver("probe", social_experiment_focus),
                _driver("window", agenda_window_focus),
                _driver("topology", topology_focus),
            )
            if item
        )
        dominant_driver = supporting_drivers[0] if supporting_drivers else ""

        arc_kind = self._arc_kind(
            person_id=person_id,
            group_thread_id=group_thread_id,
            topology_focus=topology_focus,
            commitment_target=commitment_target,
            social_experiment_focus=social_experiment_focus,
            agenda_window_focus=agenda_window_focus,
            relation_strength=relation_strength,
            trust_memory=trust_memory,
        )
        stability = self._stability(
            relation_strength=relation_strength,
            attachment=attachment,
            familiarity=familiarity,
            trust_memory=trust_memory,
            registry_total_people=registry_total_people,
            group_total_threads=group_total_threads,
            commitment_state=commitment_state,
            learning_mode_focus=learning_mode_focus,
            social_experiment_focus=social_experiment_focus,
        )
        open_tension = self._open_tension(
            agenda_window_focus=agenda_window_focus,
            agenda_window_reason=agenda_window_reason,
            topology_focus=topology_focus,
            learning_mode_focus=learning_mode_focus,
            social_experiment_focus=social_experiment_focus,
            registry_total_people=registry_total_people,
            group_total_threads=group_total_threads,
        )
        phase = self._phase(
            stability=stability,
            open_tension=open_tension,
            commitment_state=commitment_state,
            learning_mode_focus=learning_mode_focus,
            social_experiment_focus=social_experiment_focus,
            agenda_state=agenda_state,
        )
        summary = self._summary(
            arc_kind=arc_kind,
            phase=phase,
            person_id=person_id,
            group_thread_id=group_thread_id,
            social_role=social_role,
            social_interpretation=social_interpretation,
            agenda_reason=agenda_reason,
            open_tension=open_tension,
            learning_mode_focus=learning_mode_focus,
            social_experiment_focus=social_experiment_focus,
        )

        if not arc_kind and not summary and not person_id and not group_thread_id:
            return RelationArcSummary()

        return RelationArcSummary(
            arc_kind=arc_kind,
            phase=phase,
            summary=summary,
            dominant_driver=dominant_driver,
            supporting_drivers=supporting_drivers[:5],
            open_tension=open_tension,
            stability=stability,
            related_person_id=person_id,
            group_thread_id=group_thread_id,
            social_role=social_role,
            community_id=community_id,
            culture_id=culture_id,
            topology_focus=topology_focus,
            learning_mode_focus=learning_mode_focus,
            social_experiment_focus=social_experiment_focus,
        )

    def _arc_kind(
        self,
        *,
        person_id: str,
        group_thread_id: str,
        topology_focus: str,
        commitment_target: str,
        social_experiment_focus: str,
        agenda_window_focus: str,
        relation_strength: float,
        trust_memory: float,
    ) -> str:
        if commitment_target in {"repair", "bond_protect"} or social_experiment_focus == "repair_signal_probe":
            return "repairing_relation"
        if group_thread_id and topology_focus == "threaded_group":
            return "group_thread_continuity"
        if agenda_window_focus.startswith("next_") or agenda_window_focus == "long_hold":
            return "deferred_reentry"
        if person_id and relation_strength >= 0.54 and trust_memory >= 0.5:
            return "trusted_relation"
        if person_id:
            return "emerging_relation"
        return ""

    def _stability(
        self,
        *,
        relation_strength: float,
        attachment: float,
        familiarity: float,
        trust_memory: float,
        registry_total_people: int,
        group_total_threads: int,
        commitment_state: str,
        learning_mode_focus: str,
        social_experiment_focus: str,
    ) -> float:
        learning_bonus = 0.08 if learning_mode_focus == "integrate_and_commit" else 0.05 if learning_mode_focus in {"test_small", "repair_probe"} else 0.0
        probe_bonus = 0.08 if social_experiment_focus == "confirm_shared_direction" else 0.05 if social_experiment_focus in {"test_small_step", "repair_signal_probe"} else 0.0
        return _float01(
            relation_strength * 0.36
            + attachment * 0.16
            + familiarity * 0.14
            + trust_memory * 0.18
            + min(registry_total_people, 3) / 3.0 * 0.05
            + min(group_total_threads, 3) / 3.0 * 0.04
            + (0.07 if commitment_state == "commit" else 0.03 if commitment_state == "settle" else 0.0)
            + learning_bonus
            + probe_bonus
        )

    def _open_tension(
        self,
        *,
        agenda_window_focus: str,
        agenda_window_reason: str,
        topology_focus: str,
        learning_mode_focus: str,
        social_experiment_focus: str,
        registry_total_people: int,
        group_total_threads: int,
    ) -> str:
        if agenda_window_focus.startswith("next_") or agenda_window_focus == "long_hold" or agenda_window_reason == "wait_for_group_thread":
            return "timing_sensitive_reentry"
        if (
            learning_mode_focus in {"test_small", "repair_probe"}
            or social_experiment_focus in {"test_small_step", "repair_signal_probe"}
        ):
            return "careful_probe"
        if topology_focus in {"public_visible", "hierarchical"}:
            return "public_boundary"
        if registry_total_people > 1 or group_total_threads > 1:
            return "multi_thread_balance"
        return ""

    def _phase(
        self,
        *,
        stability: float,
        open_tension: str,
        commitment_state: str,
        learning_mode_focus: str,
        social_experiment_focus: str,
        agenda_state: str,
    ) -> str:
        if stability >= 0.76 and commitment_state == "commit" and not open_tension:
            return "integrating"
        if (
            agenda_state in {"repair", "revisit", "step_forward"}
            or learning_mode_focus in {"test_small", "repair_probe", "integrate_and_commit"}
            or social_experiment_focus in {"test_small_step", "repair_signal_probe", "confirm_shared_direction"}
        ) and stability >= 0.34:
            return "shifting"
        if stability >= 0.24 or open_tension:
            return "holding"
        return "forming"

    def _summary(
        self,
        *,
        arc_kind: str,
        phase: str,
        person_id: str,
        group_thread_id: str,
        social_role: str,
        social_interpretation: str,
        agenda_reason: str,
        open_tension: str,
        learning_mode_focus: str,
        social_experiment_focus: str,
    ) -> str:
        base = {
            "group_thread_continuity": "a group thread is holding across turns",
            "repairing_relation": "a relationship is being repaired in small steps",
            "deferred_reentry": "a relationship topic is being held for a better window",
            "trusted_relation": "a trusted relationship line is stabilizing",
            "emerging_relation": "a relationship line is beginning to take shape",
        }.get(arc_kind, "a relation line is forming")
        details = [base, f"phase={phase}"]
        if person_id:
            details.append(f"person={person_id}")
        if group_thread_id:
            details.append(f"group={group_thread_id}")
        if social_role:
            details.append(f"role={social_role}")
        if social_interpretation:
            details.append(f"social={social_interpretation}")
        elif agenda_reason:
            details.append(f"reason={agenda_reason}")
        if learning_mode_focus:
            details.append(f"learning={learning_mode_focus}")
        if social_experiment_focus:
            details.append(f"probe={social_experiment_focus}")
        if open_tension:
            details.append(f"tension={open_tension}")
        return " / ".join(item for item in details if item)
