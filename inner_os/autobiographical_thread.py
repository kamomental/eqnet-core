from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


AUTOBIOGRAPHICAL_THREAD_MIN_STRENGTH = 0.18
AUTOBIOGRAPHICAL_THREAD_ANCHOR_LIMIT = 160
AUTOBIOGRAPHICAL_THREAD_FOCUS_LIMIT = 120


@dataclass(frozen=True)
class AutobiographicalThreadSummary:
    mode: str = "none"
    anchor: str = ""
    focus: str = ""
    discussion_state: str = ""
    issue_state: str = ""
    residual_mode: str = ""
    strength: float = 0.0
    reason_tokens: tuple[str, ...] = field(default_factory=tuple)
    related_person_id: str = ""
    group_thread_focus: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "anchor": self.anchor,
            "focus": self.focus,
            "discussion_state": self.discussion_state,
            "issue_state": self.issue_state,
            "residual_mode": self.residual_mode,
            "strength": round(float(self.strength), 4),
            "reason_tokens": list(self.reason_tokens),
            "related_person_id": self.related_person_id,
            "group_thread_focus": self.group_thread_focus,
        }


def derive_autobiographical_thread_summary(
    current_state: Mapping[str, Any] | None,
) -> AutobiographicalThreadSummary:
    state = dict(current_state or {})
    recent_dialogue_state = dict(state.get("recent_dialogue_state") or {})
    discussion_thread_state = dict(state.get("discussion_thread_state") or {})
    issue_state = dict(state.get("issue_state") or {})
    discussion_registry = dict(state.get("discussion_thread_registry_snapshot") or {})

    anchor = _text(
        issue_state.get("issue_anchor")
        or discussion_thread_state.get("topic_anchor")
        or recent_dialogue_state.get("recent_anchor")
        or discussion_registry.get("dominant_anchor")
    )[:AUTOBIOGRAPHICAL_THREAD_ANCHOR_LIMIT]
    focus = _text(
        state.get("residual_reflection_focus")
        or state.get("current_focus")
        or state.get("long_term_theme_focus")
        or issue_state.get("issue_anchor")
        or discussion_thread_state.get("topic_anchor")
    )[:AUTOBIOGRAPHICAL_THREAD_FOCUS_LIMIT]

    discussion_state_name = _text(
        discussion_thread_state.get("state") or discussion_registry.get("dominant_issue_state")
    )
    issue_state_name = _text(issue_state.get("state"))
    residual_mode = _text(state.get("residual_reflection_mode"))
    related_person_id = _text(state.get("related_person_id"))
    group_thread_focus = _text(state.get("group_thread_focus"))

    registry_score = _registry_score(discussion_registry)
    thread_signal = _clamp01(
        registry_score * 0.42
        + _float01(recent_dialogue_state.get("thread_carry")) * 0.18
        + _float01(recent_dialogue_state.get("reopen_pressure")) * 0.08
        + _float01(discussion_thread_state.get("unresolved_pressure")) * 0.14
        + _float01(discussion_thread_state.get("revisit_readiness")) * 0.08
        + _float01(issue_state.get("pause_readiness")) * 0.06
        + _float01(issue_state.get("question_pressure")) * 0.04
    )
    residual_strength = _float01(state.get("residual_reflection_strength"))
    relationship_bonus = 0.0
    if related_person_id:
        relationship_bonus += 0.06
    if group_thread_focus:
        relationship_bonus += 0.04
    if anchor:
        relationship_bonus += 0.02
    strength = _clamp01(thread_signal * 0.68 + residual_strength * 0.24 + relationship_bonus)

    reason_tokens: list[str] = []
    if registry_score >= 0.24:
        reason_tokens.append("discussion_registry")
    if _float01(recent_dialogue_state.get("thread_carry")) >= 0.28:
        reason_tokens.append("recent_thread")
    if _float01(discussion_thread_state.get("unresolved_pressure")) >= 0.24:
        reason_tokens.append("discussion_unresolved")
    if _float01(issue_state.get("pause_readiness")) >= 0.28:
        reason_tokens.append("issue_pause")
    if residual_strength >= 0.22:
        reason_tokens.append("residual_reflection")
    if related_person_id:
        reason_tokens.append("related_person")
    if group_thread_focus:
        reason_tokens.append("group_thread")

    if strength < AUTOBIOGRAPHICAL_THREAD_MIN_STRENGTH:
        return AutobiographicalThreadSummary(
            reason_tokens=tuple(reason_tokens),
            related_person_id=related_person_id,
            group_thread_focus=group_thread_focus,
        )

    mode = "lingering_thread"
    if issue_state_name in {"pausing_issue", "exploring_issue"} or discussion_state_name in {
        "revisit_issue",
        "active_issue",
        "fresh_issue",
    }:
        mode = "unfinished_thread"
    elif residual_strength >= 0.42 and thread_signal < 0.3:
        mode = "residual_lingering"
    elif related_person_id or group_thread_focus:
        mode = "relational_lingering_thread"

    return AutobiographicalThreadSummary(
        mode=mode,
        anchor=anchor,
        focus=focus,
        discussion_state=discussion_state_name,
        issue_state=issue_state_name,
        residual_mode=residual_mode,
        strength=round(strength, 4),
        reason_tokens=tuple(reason_tokens),
        related_person_id=related_person_id,
        group_thread_focus=group_thread_focus,
    )


def _registry_score(snapshot: Mapping[str, Any]) -> float:
    thread_scores = snapshot.get("thread_scores")
    if isinstance(thread_scores, Mapping) and thread_scores:
        strongest = max((_float01(value) for value in thread_scores.values()), default=0.0)
        if strongest > 0.0:
            return strongest
    total_threads = min(int(_coerce_int(snapshot.get("total_threads"))), 4)
    uncertainty = _float01(snapshot.get("uncertainty"))
    return _clamp01(total_threads / 4.0 * 0.35 + (1.0 - uncertainty) * 0.45)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float01(value: Any) -> float:
    try:
        numeric = float(value or 0.0)
    except (TypeError, ValueError):
        numeric = 0.0
    return _clamp01(numeric)


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value or 0.0)
    except (TypeError, ValueError):
        numeric = 0.0
    if numeric <= 0.0:
        return 0.0
    if numeric >= 1.0:
        return 1.0
    return float(numeric)
