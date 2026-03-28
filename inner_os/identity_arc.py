from __future__ import annotations

from dataclasses import dataclass, field
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
class IdentityArcSummary:
    summary_version: str = "v1"
    arc_kind: str = ""
    phase: str = "forming"
    summary: str = ""
    dominant_driver: str = ""
    supporting_drivers: tuple[str, ...] = ()
    open_tension: str = ""
    stability: float = 0.0
    memory_anchor: str = ""
    related_person_id: str = ""
    group_thread_focus: str = ""
    long_term_theme_kind: str = ""
    long_term_theme_focus: str = ""
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
            "memory_anchor": self.memory_anchor,
            "related_person_id": self.related_person_id,
            "group_thread_focus": self.group_thread_focus,
            "long_term_theme_kind": self.long_term_theme_kind,
            "long_term_theme_focus": self.long_term_theme_focus,
            "learning_mode_focus": self.learning_mode_focus,
            "social_experiment_focus": self.social_experiment_focus,
        }


class IdentityArcSummaryBuilder:
    def build(self, report: Mapping[str, Any] | None) -> IdentityArcSummary:
        payload = dict(report or {})
        memory_summary = dict(payload.get("inner_os_memory_class_summary") or {})
        agenda_summary = dict(payload.get("inner_os_agenda_summary") or {})
        commitment_summary = dict(payload.get("inner_os_commitment_summary") or {})
        insight_summary = dict(payload.get("inner_os_insight_summary") or {})
        partner_summary = dict(payload.get("inner_os_partner_relation_summary") or {})
        group_thread_summary = dict(payload.get("inner_os_group_thread_registry_summary") or {})
        theme_summary = dict(payload.get("inner_os_long_term_theme_summary") or {})

        theme_kind = _text(theme_summary.get("kind"))
        theme_focus = _text(theme_summary.get("focus"))
        theme_anchor = _text(theme_summary.get("anchor"))
        theme_summary_text = _text(theme_summary.get("summary"))
        theme_strength = _float01(theme_summary.get("strength"))

        memory_class = _text(memory_summary.get("dominant_class"))
        agenda_state = _text(payload.get("inner_os_same_turn_agenda_state")) or _text(agenda_summary.get("dominant_agenda"))
        agenda_reason = _text(payload.get("inner_os_same_turn_agenda_reason")) or _text(agenda_summary.get("dominant_reason"))
        agenda_bias = _float01(agenda_summary.get("agenda_carry_bias"))
        agenda_window_focus = _text(payload.get("inner_os_sleep_agenda_window_focus"))
        learning_mode_focus = _text(payload.get("inner_os_same_turn_learning_mode_state")) or _text(payload.get("inner_os_sleep_learning_mode_focus"))
        social_experiment_focus = _text(payload.get("inner_os_same_turn_social_experiment_state")) or _text(payload.get("inner_os_sleep_social_experiment_focus"))
        commitment_target = _text(commitment_summary.get("dominant_target"))
        commitment_state = _text(commitment_summary.get("dominant_state")) or "waver"
        commitment_reason = _text(commitment_summary.get("dominant_reason"))
        commitment_bias = _float01(commitment_summary.get("commitment_carry_bias"))
        insight_class = _text(insight_summary.get("dominant_insight_class"))
        insight_topic = _text(insight_summary.get("dominant_reframed_topic"))
        insight_reframe_bias = _float01(insight_summary.get("insight_reframing_bias"))
        related_person_id = _text(partner_summary.get("person_id"))
        partner_strength = _float01(partner_summary.get("strength"))
        memory_anchor = _text(partner_summary.get("memory_anchor")) or theme_anchor
        group_thread_focus = _text(group_thread_summary.get("dominant_thread_id"))
        group_thread_total = int(group_thread_summary.get("total_threads") or 0)

        supporting_drivers = tuple(
            driver
            for driver in (
                _driver("theme", theme_kind or theme_focus),
                _driver("memory", memory_class),
                _driver("agenda", agenda_state),
                _driver("learning", learning_mode_focus),
                _driver("probe", social_experiment_focus),
                _driver("commitment", commitment_target or commitment_state),
                _driver("insight", insight_class),
                _driver("relation", related_person_id),
                _driver("thread", group_thread_focus),
            )
            if driver
        )
        dominant_driver = supporting_drivers[0] if supporting_drivers else ""

        arc_kind = self._arc_kind(
            memory_class=memory_class,
            agenda_state=agenda_state,
            commitment_target=commitment_target,
            insight_class=insight_class,
            theme_kind=theme_kind,
            related_person_id=related_person_id,
            group_thread_focus=group_thread_focus,
            learning_mode_focus=learning_mode_focus,
            social_experiment_focus=social_experiment_focus,
        )
        stability = self._stability(
            theme_strength=theme_strength,
            partner_strength=partner_strength,
            agenda_bias=agenda_bias,
            commitment_bias=commitment_bias,
            insight_reframe_bias=insight_reframe_bias,
            group_thread_total=group_thread_total,
            commitment_state=commitment_state,
            learning_mode_focus=learning_mode_focus,
            social_experiment_focus=social_experiment_focus,
        )
        open_tension = self._open_tension(
            memory_class=memory_class,
            agenda_window_focus=agenda_window_focus,
            commitment_target=commitment_target,
            commitment_state=commitment_state,
            related_person_id=related_person_id,
            group_thread_total=group_thread_total,
            learning_mode_focus=learning_mode_focus,
            social_experiment_focus=social_experiment_focus,
        )
        phase = self._phase(
            stability=stability,
            agenda_state=agenda_state,
            commitment_target=commitment_target,
            commitment_state=commitment_state,
            insight_class=insight_class,
            open_tension=open_tension,
            learning_mode_focus=learning_mode_focus,
            social_experiment_focus=social_experiment_focus,
        )
        summary = self._summary(
            arc_kind=arc_kind,
            phase=phase,
            memory_anchor=memory_anchor,
            related_person_id=related_person_id,
            group_thread_focus=group_thread_focus,
            theme_summary=theme_summary_text,
            agenda_reason=agenda_reason,
            commitment_reason=commitment_reason,
            insight_topic=insight_topic,
            open_tension=open_tension,
            learning_mode_focus=learning_mode_focus,
            social_experiment_focus=social_experiment_focus,
        )

        if not arc_kind and not summary and not memory_anchor and not theme_focus:
            return IdentityArcSummary()

        return IdentityArcSummary(
            arc_kind=arc_kind,
            phase=phase,
            summary=summary,
            dominant_driver=dominant_driver,
            supporting_drivers=supporting_drivers[:5],
            open_tension=open_tension,
            stability=stability,
            memory_anchor=memory_anchor,
            related_person_id=related_person_id,
            group_thread_focus=group_thread_focus,
            long_term_theme_kind=theme_kind,
            long_term_theme_focus=theme_focus,
            learning_mode_focus=learning_mode_focus,
            social_experiment_focus=social_experiment_focus,
        )

    def _arc_kind(
        self,
        *,
        memory_class: str,
        agenda_state: str,
        commitment_target: str,
        insight_class: str,
        theme_kind: str,
        related_person_id: str,
        group_thread_focus: str,
        learning_mode_focus: str,
        social_experiment_focus: str,
    ) -> str:
        if commitment_target in {"repair", "bond_protect"} or memory_class == "bond_protection":
            if related_person_id or group_thread_focus:
                return "repairing_bond"
            return "repairing_self"
        if agenda_state in {"hold", "revisit"} and (related_person_id or group_thread_focus):
            return "holding_thread"
        if social_experiment_focus == "repair_signal_probe" and (related_person_id or group_thread_focus):
            return "repairing_bond"
        if theme_kind in {"meaning", "identity"} and insight_class in {"reframed_relation", "insight_trace"}:
            return "reframing_path"
        if theme_kind in {"place", "ritual"} and (related_person_id or group_thread_focus):
            return "shared_place_thread"
        if commitment_target in {"hold", "stabilize"} or memory_class in {"body_risk", "unresolved_tension"}:
            return "stabilizing_self"
        if (
            agenda_state in {"step_forward", "revisit"}
            or learning_mode_focus in {"test_small", "repair_probe", "integrate_and_commit"}
            or social_experiment_focus in {"test_small_step", "confirm_shared_direction"}
        ):
            return "growing_edge"
        if theme_kind or related_person_id or group_thread_focus:
            return "emerging_continuity"
        return ""

    def _stability(
        self,
        *,
        theme_strength: float,
        partner_strength: float,
        agenda_bias: float,
        commitment_bias: float,
        insight_reframe_bias: float,
        group_thread_total: int,
        commitment_state: str,
        learning_mode_focus: str,
        social_experiment_focus: str,
    ) -> float:
        learning_bonus = 0.08 if learning_mode_focus == "integrate_and_commit" else 0.05 if learning_mode_focus in {"test_small", "repair_probe"} else 0.0
        probe_bonus = 0.07 if social_experiment_focus == "confirm_shared_direction" else 0.05 if social_experiment_focus in {"test_small_step", "repair_signal_probe"} else 0.0
        score = (
            theme_strength * 0.28
            + partner_strength * 0.18
            + agenda_bias * 0.14
            + commitment_bias * 0.2
            + insight_reframe_bias * 0.08
            + min(group_thread_total, 3) / 3.0 * 0.06
            + (0.08 if commitment_state == "commit" else 0.04 if commitment_state == "settle" else 0.0)
            + learning_bonus
            + probe_bonus
        )
        return _float01(score)

    def _open_tension(
        self,
        *,
        memory_class: str,
        agenda_window_focus: str,
        commitment_target: str,
        commitment_state: str,
        related_person_id: str,
        group_thread_total: int,
        learning_mode_focus: str,
        social_experiment_focus: str,
    ) -> str:
        if memory_class == "unresolved_tension":
            return "unresolved_tension"
        if agenda_window_focus.startswith("next_") or agenda_window_focus == "long_hold":
            return "timing_sensitive_reentry"
        if (
            learning_mode_focus in {"test_small", "repair_probe"}
            or social_experiment_focus in {"test_small_step", "repair_signal_probe"}
        ):
            return "careful_probe"
        if commitment_target in {"repair", "bond_protect"} and commitment_state != "commit":
            return "careful_repair"
        if group_thread_total > 1:
            return "multi_thread_balance"
        if related_person_id and commitment_target in {"hold", "stabilize"}:
            return "guarded_closeness"
        return ""

    def _phase(
        self,
        *,
        stability: float,
        agenda_state: str,
        commitment_target: str,
        commitment_state: str,
        insight_class: str,
        open_tension: str,
        learning_mode_focus: str,
        social_experiment_focus: str,
    ) -> str:
        if stability >= 0.76 and commitment_state == "commit" and not open_tension:
            return "integrating"
        if (
            insight_class in {"reframed_relation", "insight_trace"}
            or agenda_state in {"step_forward", "repair"}
            or commitment_target in {"repair", "bond_protect", "step_forward"}
            or learning_mode_focus in {"test_small", "repair_probe", "integrate_and_commit"}
            or social_experiment_focus in {"test_small_step", "repair_signal_probe", "confirm_shared_direction"}
        ) and stability >= 0.34:
            return "shifting"
        if stability >= 0.26 or commitment_state in {"settle", "commit"}:
            return "holding"
        return "forming"

    def _summary(
        self,
        *,
        arc_kind: str,
        phase: str,
        memory_anchor: str,
        related_person_id: str,
        group_thread_focus: str,
        theme_summary: str,
        agenda_reason: str,
        commitment_reason: str,
        insight_topic: str,
        open_tension: str,
        learning_mode_focus: str,
        social_experiment_focus: str,
    ) -> str:
        base = {
            "repairing_bond": "repair is gathering around a relationship thread",
            "repairing_self": "repair is gathering around a fragile internal thread",
            "holding_thread": "a relational thread is being kept warm without forcing reentry",
            "reframing_path": "a reframed meaning is beginning to cohere",
            "shared_place_thread": "a shared place or ritual is holding continuity together",
            "stabilizing_self": "stabilization is protecting continuity before further movement",
            "growing_edge": "a next step is starting to hold shape",
            "emerging_continuity": "a continuity line is beginning to take form",
        }.get(arc_kind, "a continuity line is forming")
        details: list[str] = [base, f"phase={phase}"]
        if memory_anchor:
            details.append(f"anchor={memory_anchor}")
        if related_person_id:
            details.append(f"person={related_person_id}")
        elif group_thread_focus:
            details.append(f"thread={group_thread_focus}")
        if theme_summary:
            details.append(f"theme={theme_summary}")
        elif insight_topic:
            details.append(f"topic={insight_topic}")
        elif commitment_reason:
            details.append(f"reason={commitment_reason}")
        elif agenda_reason:
            details.append(f"reason={agenda_reason}")
        if learning_mode_focus:
            details.append(f"learning={learning_mode_focus}")
        if social_experiment_focus:
            details.append(f"probe={social_experiment_focus}")
        if open_tension:
            details.append(f"tension={open_tension}")
        return " / ".join(item for item in details if item)
