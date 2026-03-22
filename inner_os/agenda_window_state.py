from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class AgendaWindowState:
    state: str
    reason: str
    score: float
    scores: dict[str, float]
    winner_margin: float
    dominant_inputs: list[str]
    deferral_budget: float
    carry_target: str
    opportunistic_ok: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "reason": self.reason,
            "score": round(self.score, 4),
            "scores": {key: round(value, 4) for key, value in self.scores.items()},
            "winner_margin": round(self.winner_margin, 4),
            "dominant_inputs": list(self.dominant_inputs),
            "deferral_budget": round(self.deferral_budget, 4),
            "carry_target": self.carry_target,
            "opportunistic_ok": bool(self.opportunistic_ok),
        }


def derive_agenda_window_state(
    *,
    self_state: Mapping[str, Any],
    agenda_state: Mapping[str, Any],
    body_recovery_guard: Mapping[str, Any],
    body_homeostasis_state: Mapping[str, Any],
    homeostasis_budget_state: Mapping[str, Any],
    initiative_followup_bias: Mapping[str, Any],
    commitment_state: Mapping[str, Any],
    relational_continuity_state: Mapping[str, Any],
    relation_competition_state: Mapping[str, Any],
    social_topology_state: Mapping[str, Any],
    cultural_conversation_state: Mapping[str, Any],
    related_person_ids: Sequence[str],
    relation_bias_strength: float,
    scene_family: str = "",
) -> AgendaWindowState:
    state_payload = dict(self_state or {})
    agenda = dict(agenda_state or {})
    body_guard = dict(body_recovery_guard or {})
    body_homeostasis = dict(body_homeostasis_state or {})
    budget = dict(homeostasis_budget_state or {})
    followup = dict(initiative_followup_bias or {})
    commitment = dict(commitment_state or {})
    continuity = dict(relational_continuity_state or {})
    competition = dict(relation_competition_state or {})
    topology = dict(social_topology_state or {})
    culture = dict(cultural_conversation_state or {})

    agenda_name = str(agenda.get("state") or "hold").strip() or "hold"
    agenda_score = _clamp01(agenda.get("score"))
    agenda_reason = str(agenda.get("reason") or "").strip()
    guard_name = str(body_guard.get("state") or "open").strip() or "open"
    guard_score = _clamp01(body_guard.get("score"))
    body_name = str(body_homeostasis.get("state") or "steady").strip() or "steady"
    body_score = _clamp01(body_homeostasis.get("score"))
    budget_name = str(budget.get("state") or "steady").strip() or "steady"
    budget_score = _clamp01(budget.get("score"))
    followup_name = str(followup.get("state") or "hold").strip() or "hold"
    followup_score = _clamp01(followup.get("score"))
    commitment_target = str(commitment.get("target") or "hold").strip() or "hold"
    commitment_score = _clamp01(commitment.get("score"))
    continuity_name = str(continuity.get("state") or "distant").strip() or "distant"
    continuity_score = _clamp01(continuity.get("score"))
    topology_name = str(topology.get("state") or "ambient").strip() or "ambient"
    topology_score = _clamp01(topology.get("score"))
    threading_pressure = _clamp01(topology.get("threading_pressure"))
    visibility_pressure = _clamp01(topology.get("visibility_pressure"))
    hierarchy_pressure = _clamp01(topology.get("hierarchy_pressure"))
    culture_name = str(culture.get("state") or "careful_polite").strip() or "careful_polite"
    directness_ceiling = _clamp01(culture.get("directness_ceiling"))
    politeness_pressure = _clamp01(culture.get("politeness_pressure"))

    semantic_strength = _clamp01(state_payload.get("semantic_seed_strength"))
    semantic_recurrence = _clamp01(float(state_payload.get("semantic_seed_recurrence", 0.0) or 0.0) / 2.0)
    prospective_memory_pull = _clamp01(state_payload.get("prospective_memory_pull"))
    pending_meaning = _clamp01(state_payload.get("pending_meaning"))
    stress = _clamp01(state_payload.get("stress"))
    recovery_need = _clamp01(state_payload.get("recovery_need"))

    carry_focus = str(state_payload.get("agenda_window_focus") or "").strip()
    carry_bias = _clamp01(state_payload.get("agenda_window_bias"))
    related_people = [str(item).strip() for item in related_person_ids if str(item).strip()]
    relation_active = 1.0 if (relation_bias_strength >= 0.28 or bool(related_people)) else 0.0
    private_room = 1.0 if topology_name in {"one_to_one", "ambient"} and visibility_pressure <= 0.24 and hierarchy_pressure <= 0.24 else 0.0
    group_room = 1.0 if topology_name == "threaded_group" or threading_pressure >= 0.28 else 0.0
    public_pressure = max(visibility_pressure, hierarchy_pressure, politeness_pressure)
    same_culture_pull = _clamp01(semantic_strength * 0.46 + semantic_recurrence * 0.34 + (1.0 - directness_ceiling) * 0.2)
    similar_person_room = _clamp01(
        semantic_recurrence * 0.42
        + prospective_memory_pull * 0.26
        + (0.18 if continuity_name in {"holding_thread", "reopening"} else 0.0)
        + (0.14 if culture_name in {"group_attuned", "public_courteous", "hierarchy_respectful"} else 0.0)
    )

    now_score = _clamp01(
        agenda_score * 0.22
        + (0.2 if agenda_name in {"step_forward", "repair"} else 0.0)
        + private_room * 0.18
        + (0.12 if continuity_name in {"reopening", "co_regulating"} else 0.0) * max(continuity_score, 0.5)
        + (0.1 if followup_name == "offer_next_step" else 0.0) * max(followup_score, 0.5)
        - guard_score * 0.16
        - max(body_score, budget_score) * 0.16
        - public_pressure * 0.12
    )
    next_private_score = _clamp01(
        agenda_score * 0.16
        + relation_active * 0.14
        + public_pressure * 0.22
        + (0.14 if topology_name in {"public_visible", "hierarchical"} else 0.0)
        + (0.1 if agenda_name in {"repair", "revisit"} else 0.0)
        + (0.08 if scene_family in {"reverent_distance", "guarded_boundary"} else 0.0)
    )
    next_same_group_score = _clamp01(
        agenda_score * 0.14
        + group_room * 0.22
        + threading_pressure * 0.18
        + topology_score * 0.08
        + (0.12 if agenda_name in {"repair", "revisit"} else 0.0)
        + (0.08 if continuity_name in {"holding_thread", "reopening"} else 0.0)
    )
    next_same_culture_score = _clamp01(
        agenda_score * 0.1
        + same_culture_pull * 0.28
        + semantic_strength * 0.12
        + semantic_recurrence * 0.12
        + (0.1 if culture_name in {"group_attuned", "public_courteous", "hierarchy_respectful"} else 0.0)
        + (0.08 if relation_active <= 0.0 else 0.0)
    )
    opportunistic_reentry_score = _clamp01(
        agenda_score * 0.1
        + similar_person_room * 0.28
        + prospective_memory_pull * 0.14
        + pending_meaning * 0.12
        + (0.1 if agenda_name in {"revisit", "repair"} else 0.0)
        + (0.08 if followup_name == "reopen_softly" else 0.0) * max(followup_score, 0.5)
        - public_pressure * 0.08
    )
    long_hold_score = _clamp01(
        guard_score * 0.22
        + max(body_score, budget_score) * 0.2
        + recovery_need * 0.12
        + stress * 0.1
        + (0.1 if agenda_name == "hold" else 0.0)
        + (0.08 if commitment_target in {"hold", "stabilize"} else 0.0) * max(commitment_score, 0.5)
    )

    if guard_name == "recovery_first" or body_name == "depleted" or budget_name == "depleted":
        long_hold_score = _clamp01(long_hold_score + 0.24)
        now_score *= 0.25
        opportunistic_reentry_score *= 0.52
    elif guard_name == "guarded" or body_name == "recovering" or budget_name == "recovering":
        now_score *= 0.64
        next_private_score = _clamp01(next_private_score + 0.06)

    if carry_focus and carry_bias > 0.0:
        if carry_focus == "now":
            now_score = _clamp01(now_score + carry_bias * 0.18)
        elif carry_focus == "next_private_window":
            next_private_score = _clamp01(next_private_score + carry_bias * 0.18)
        elif carry_focus == "next_same_group_window":
            next_same_group_score = _clamp01(next_same_group_score + carry_bias * 0.18)
        elif carry_focus == "next_same_culture_window":
            next_same_culture_score = _clamp01(next_same_culture_score + carry_bias * 0.18)
        elif carry_focus == "opportunistic_reentry":
            opportunistic_reentry_score = _clamp01(opportunistic_reentry_score + carry_bias * 0.18)
        elif carry_focus == "long_hold":
            long_hold_score = _clamp01(long_hold_score + carry_bias * 0.18)

    scores = {
        "now": now_score,
        "next_private_window": next_private_score,
        "next_same_group_window": next_same_group_score,
        "next_same_culture_window": next_same_culture_score,
        "opportunistic_reentry": opportunistic_reentry_score,
        "long_hold": long_hold_score,
    }
    winner, margin = _winner_and_margin(scores)
    if scores.get(winner, 0.0) <= 0.0:
        winner = "long_hold"
        margin = 0.0

    reason = _reason_for_window(
        winner=winner,
        guard_name=guard_name,
        agenda_name=agenda_name,
        agenda_reason=agenda_reason,
        carry_reason=str(state_payload.get("agenda_window_reason") or "").strip(),
    )
    deferral_budget = _deferral_budget(
        winner=winner,
        semantic_recurrence=semantic_recurrence,
        prospective_memory_pull=prospective_memory_pull,
        agenda_score=agenda_score,
    )
    carry_target = _carry_target_for_window(winner, relation_active=relation_active, topology_name=topology_name)
    dominant_inputs = _compact(
        [
            "agenda_state" if agenda_score >= 0.22 else "",
            "body_recovery_guard" if guard_name in {"guarded", "recovery_first"} else "",
            "body_homeostasis_state" if body_name in {"recovering", "depleted"} and body_score >= 0.18 else "",
            "homeostasis_budget_state" if budget_name in {"recovering", "depleted"} and budget_score >= 0.16 else "",
            "initiative_followup_bias" if followup_score >= 0.16 and followup_name in {"reopen_softly", "offer_next_step"} else "",
            "relational_continuity" if continuity_score >= 0.22 and continuity_name in {"holding_thread", "reopening", "co_regulating"} else "",
            "social_topology" if topology_score >= 0.2 or topology_name in {"threaded_group", "public_visible", "hierarchical"} else "",
            "cultural_conversation" if same_culture_pull >= 0.24 or culture_name in {"group_attuned", "public_courteous", "hierarchy_respectful"} else "",
            "semantic_seed" if semantic_strength >= 0.24 else "",
            "semantic_recurrence" if semantic_recurrence >= 0.2 else "",
            "prospective_memory_pull" if prospective_memory_pull >= 0.18 else "",
            "relation_competition" if str(competition.get("state") or "").strip() not in {"", "ambient"} else "",
            "overnight_agenda_window_carry" if carry_focus and carry_bias > 0.0 else "",
        ]
    )
    return AgendaWindowState(
        state=winner,
        reason=reason,
        score=_clamp01(scores.get(winner, 0.0)),
        scores=scores,
        winner_margin=margin,
        dominant_inputs=dominant_inputs,
        deferral_budget=deferral_budget,
        carry_target=carry_target,
        opportunistic_ok=winner in {"opportunistic_reentry", "next_same_culture_window"} and deferral_budget >= 0.22,
    )


def _reason_for_window(
    *,
    winner: str,
    guard_name: str,
    agenda_name: str,
    agenda_reason: str,
    carry_reason: str,
) -> str:
    if winner == "long_hold":
        if guard_name == "recovery_first":
            return "recovery_first"
        return "hold_until_safer_time"
    if winner == "now":
        if agenda_name == "step_forward":
            return "step_forward_now"
        if agenda_name == "repair":
            return "repair_window_open"
        return agenda_reason or "open_window"
    if winner == "next_private_window":
        return "wait_for_private_window"
    if winner == "next_same_group_window":
        return "wait_for_group_thread"
    if winner == "next_same_culture_window":
        return "wait_for_same_culture_window"
    return carry_reason or "similar_person_reentry"


def _deferral_budget(
    *,
    winner: str,
    semantic_recurrence: float,
    prospective_memory_pull: float,
    agenda_score: float,
) -> float:
    if winner == "now":
        return _clamp01(agenda_score * 0.12)
    if winner == "next_private_window":
        return _clamp01(0.28 + prospective_memory_pull * 0.24 + agenda_score * 0.1)
    if winner == "next_same_group_window":
        return _clamp01(0.36 + semantic_recurrence * 0.24 + agenda_score * 0.08)
    if winner == "next_same_culture_window":
        return _clamp01(0.44 + semantic_recurrence * 0.24 + prospective_memory_pull * 0.12)
    if winner == "opportunistic_reentry":
        return _clamp01(0.32 + semantic_recurrence * 0.18 + prospective_memory_pull * 0.22)
    return _clamp01(0.52 + semantic_recurrence * 0.12)


def _carry_target_for_window(winner: str, *, relation_active: float, topology_name: str) -> str:
    if winner == "now":
        return "current_window"
    if winner == "next_private_window":
        return "same_person_private_window" if relation_active >= 1.0 else "next_private_window"
    if winner == "next_same_group_window":
        return "same_group_thread"
    if winner == "next_same_culture_window":
        return "same_culture_window"
    if winner == "opportunistic_reentry":
        return "similar_person_or_theme"
    if topology_name == "threaded_group":
        return "same_group_thread"
    return "later_safe_window"


def _winner_and_margin(scores: Mapping[str, float]) -> tuple[str, float]:
    ranked = sorted(
        ((str(key), _clamp01(value)) for key, value in scores.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ranked:
        return "long_hold", 0.0
    winner_key, winner_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    return winner_key, _clamp01(winner_score - runner_up)


def _compact(values: list[str]) -> list[str]:
    ordered: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in ordered:
            ordered.append(text)
    return ordered


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))
