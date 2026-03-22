from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class AgendaState:
    state: str
    reason: str
    score: float
    scores: dict[str, float]
    winner_margin: float
    dominant_inputs: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "reason": self.reason,
            "score": round(self.score, 4),
            "scores": {key: round(value, 4) for key, value in self.scores.items()},
            "winner_margin": round(self.winner_margin, 4),
            "dominant_inputs": list(self.dominant_inputs),
        }


def derive_agenda_state(
    *,
    self_state: Mapping[str, Any],
    body_recovery_guard: Mapping[str, Any],
    body_homeostasis_state: Mapping[str, Any],
    homeostasis_budget_state: Mapping[str, Any],
    initiative_readiness: Mapping[str, Any],
    initiative_followup_bias: Mapping[str, Any],
    commitment_state: Mapping[str, Any],
    protection_mode: Mapping[str, Any],
    memory_write_class: str,
    insight_event: Mapping[str, Any] | None = None,
) -> AgendaState:
    self_payload = dict(self_state or {})
    body_guard = dict(body_recovery_guard or {})
    body_homeostasis = dict(body_homeostasis_state or {})
    budget = dict(homeostasis_budget_state or {})
    readiness = dict(initiative_readiness or {})
    followup = dict(initiative_followup_bias or {})
    commitment = dict(commitment_state or {})
    protection = dict(protection_mode or {})
    insight = dict(insight_event or {})

    stress = _clamp01(float(self_payload.get("stress", 0.0) or 0.0))
    recovery_need = _clamp01(float(self_payload.get("recovery_need", 0.0) or 0.0))
    pending_meaning = _clamp01(float(self_payload.get("pending_meaning", 0.0) or 0.0))
    unresolved_count = _clamp01(float(self_payload.get("unresolved_count", 0.0) or 0.0) / 4.0)
    prospective_memory_pull = _clamp01(float(self_payload.get("prospective_memory_pull", 0.0) or 0.0))
    recent_strain = _clamp01(float(self_payload.get("recent_strain", 0.0) or 0.0))
    degraded = _clamp01(
        float(self_payload.get("degraded", 0.0) or 0.0)
        if not isinstance(self_payload.get("degraded"), bool)
        else (1.0 if self_payload.get("degraded") else 0.0)
    )

    guard_state = str(body_guard.get("state") or "open").strip() or "open"
    guard_score = _clamp01(float(body_guard.get("score", 0.0) or 0.0))
    body_state = str(body_homeostasis.get("state") or "steady").strip() or "steady"
    body_score = _clamp01(float(body_homeostasis.get("score", 0.0) or 0.0))
    budget_state = str(budget.get("state") or "steady").strip() or "steady"
    budget_score = _clamp01(float(budget.get("score", 0.0) or 0.0))
    reserve_level = _clamp01(float(budget.get("reserve_level", 0.0) or 0.0))
    debt_level = _clamp01(float(budget.get("debt_level", 0.0) or 0.0))

    readiness_state = str(readiness.get("state") or "hold").strip() or "hold"
    readiness_score = _clamp01(float(readiness.get("score", 0.0) or 0.0))
    followup_state = str(followup.get("state") or "hold").strip() or "hold"
    followup_score = _clamp01(float(followup.get("score", 0.0) or 0.0))

    commitment_target = str(commitment.get("target") or "hold").strip() or "hold"
    commitment_mode = str(commitment.get("state") or "waver").strip() or "waver"
    commitment_score = _clamp01(float(commitment.get("score", 0.0) or 0.0))

    protection_mode = str(protection.get("mode") or "").strip()
    protection_strength = _clamp01(float(protection.get("strength", 0.0) or 0.0))

    insight_triggered = 1.0 if bool(insight.get("triggered", False)) else 0.0
    insight_orient_bias = _clamp01(
        float(insight.get("orient_bias", 0.0) or 0.0)
        if not isinstance(insight.get("orient_bias"), Mapping)
        else 0.0
    )
    if insight_orient_bias <= 0.0:
        insight_orient_bias = _clamp01(
            float((insight.get("score") or {}).get("tension_relief", 0.0) or 0.0) * 0.6
            + float((insight.get("score") or {}).get("novelty_gain", 0.0) or 0.0) * 0.2
        )

    memory_class = str(memory_write_class or "").strip()
    repair_memory = 1.0 if memory_class == "repair_trace" else 0.0
    bond_memory = 1.0 if memory_class == "bond_protection" else 0.0
    unresolved_memory = 1.0 if memory_class == "unresolved_tension" else 0.0
    insight_memory = 1.0 if memory_class == "insight_trace" else 0.0

    hold_score = _clamp01(
        0.26 * guard_score
        + 0.16 * max(body_score, budget_score)
        + 0.12 * debt_level
        + 0.12 * recovery_need
        + 0.1 * stress
        + 0.1 * degraded
        + 0.08 * (1.0 if protection_mode in {"contain", "stabilize", "shield"} else 0.0) * protection_strength
        + 0.06 * (1.0 if followup_state == "hold" else 0.0) * followup_score
    )
    revisit_score = _clamp01(
        0.22 * pending_meaning
        + 0.18 * unresolved_count
        + 0.14 * prospective_memory_pull
        + 0.12 * (1.0 if followup_state == "reopen_softly" else 0.0) * followup_score
        + 0.1 * insight_triggered
        + 0.08 * insight_orient_bias
        + 0.08 * unresolved_memory
        + 0.06 * insight_memory
        + 0.06 * (1.0 if commitment_target == "hold" and commitment_mode != "commit" else 0.0)
        - 0.08 * degraded
    )
    repair_score = _clamp01(
        0.24 * (1.0 if commitment_target in {"repair", "bond_protect"} else 0.0) * max(commitment_score, 0.5)
        + 0.18 * (1.0 if protection_mode == "repair" else 0.0) * protection_strength
        + 0.12 * (1.0 if followup_state == "reopen_softly" else 0.0) * followup_score
        + 0.12 * repair_memory
        + 0.08 * bond_memory
        + 0.08 * prospective_memory_pull
        + 0.08 * insight_orient_bias
        + 0.06 * (1.0 if readiness_state in {"tentative", "ready"} else 0.0) * readiness_score
        - 0.1 * degraded
        - 0.08 * debt_level
    )
    step_forward_score = _clamp01(
        0.24 * (1.0 if readiness_state == "ready" else 0.0) * readiness_score
        + 0.18 * (1.0 if followup_state == "offer_next_step" else 0.0) * followup_score
        + 0.16 * (1.0 if commitment_target == "step_forward" else 0.0) * max(commitment_score, 0.5)
        + 0.12 * prospective_memory_pull
        + 0.08 * max(0.0, 1.0 - recent_strain)
        + 0.08 * reserve_level
        - 0.12 * guard_score
        - 0.1 * max(body_score, budget_score)
        - 0.08 * degraded
    )

    hard_hold = 1.0 if (
        guard_state == "recovery_first"
        or protection_mode == "shield"
        or body_state == "depleted"
        or budget_state == "depleted"
    ) else 0.0
    if hard_hold >= 1.0:
        hold_score = _clamp01(hold_score + 0.24)
        revisit_score = _clamp01(revisit_score * 0.76)
        repair_score = _clamp01(repair_score * 0.48)
        step_forward_score = _clamp01(step_forward_score * 0.25)
    elif guard_state == "guarded" or protection_mode == "stabilize":
        repair_score = _clamp01(repair_score * 0.74)
        step_forward_score = _clamp01(step_forward_score * 0.52)

    scores = {
        "hold": hold_score,
        "revisit": revisit_score,
        "repair": repair_score,
        "step_forward": step_forward_score,
    }
    state, winner_margin = _winner_and_margin(scores)
    reason = _reason_for_state(
        state=state,
        guard_state=guard_state,
        protection_mode=protection_mode,
        followup_state=followup_state,
        commitment_target=commitment_target,
        memory_class=memory_class,
        insight_triggered=insight_triggered >= 1.0,
        pending_meaning=pending_meaning,
        prospective_memory_pull=prospective_memory_pull,
    )
    dominant_inputs = _compact(
        [
            "body_recovery_guard" if guard_state in {"guarded", "recovery_first"} else "",
            "body_homeostasis_state" if body_state in {"strained", "recovering", "depleted"} and body_score >= 0.18 else "",
            "homeostasis_budget_state" if budget_state in {"strained", "recovering", "depleted"} and budget_score >= 0.16 else "",
            "initiative_readiness" if readiness_state in {"tentative", "ready"} and readiness_score >= 0.2 else "",
            "initiative_followup_bias" if followup_score >= 0.16 and followup_state in {"reopen_softly", "offer_next_step", "hold"} else "",
            "commitment_state" if commitment_score >= 0.24 and commitment_target in {"hold", "repair", "bond_protect", "step_forward"} else "",
            "protection_mode" if protection_strength >= 0.22 and protection_mode in {"contain", "stabilize", "repair", "shield"} else "",
            "pending_meaning" if pending_meaning >= 0.2 else "",
            "prospective_memory_pull" if prospective_memory_pull >= 0.18 else "",
            "insight_event" if insight_triggered >= 1.0 else "",
            "repair_trace" if repair_memory >= 1.0 else "",
            "bond_protection" if bond_memory >= 1.0 else "",
            "unresolved_tension" if unresolved_memory >= 1.0 else "",
            "insight_trace" if insight_memory >= 1.0 else "",
        ]
    )
    return AgendaState(
        state=state,
        reason=reason,
        score=_clamp01(float(scores.get(state, 0.0) or 0.0)),
        scores=scores,
        winner_margin=winner_margin,
        dominant_inputs=dominant_inputs,
    )


def _reason_for_state(
    *,
    state: str,
    guard_state: str,
    protection_mode: str,
    followup_state: str,
    commitment_target: str,
    memory_class: str,
    insight_triggered: bool,
    pending_meaning: float,
    prospective_memory_pull: float,
) -> str:
    if state == "hold":
        if guard_state == "recovery_first":
            return "recovery_first"
        if protection_mode == "shield":
            return "shield"
        return "hold_open"
    if state == "repair":
        if commitment_target in {"repair", "bond_protect"}:
            return commitment_target
        if memory_class in {"repair_trace", "bond_protection"}:
            return memory_class
        return "repair_window"
    if state == "step_forward":
        if commitment_target == "step_forward":
            return "step_forward"
        if followup_state == "offer_next_step":
            return "offer_next_step"
        return "forward_pull"
    if insight_triggered and pending_meaning >= 0.16:
        return "insight_revisit"
    if followup_state == "reopen_softly":
        return "reopen_softly"
    if prospective_memory_pull >= 0.18:
        return "prospective_revisit"
    return "revisit_open_loop"


def _winner_and_margin(scores: Mapping[str, float]) -> tuple[str, float]:
    ranked = sorted(
        ((str(key), _clamp01(float(value or 0.0))) for key, value in scores.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ranked:
        return "hold", 0.0
    winner_key, winner_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    return winner_key, _clamp01(winner_score - runner_up)


def _compact(values: list[str]) -> list[str]:
    compacted: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in compacted:
            compacted.append(text)
    return compacted


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
