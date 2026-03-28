from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class LearningModeState:
    state: str
    score: float
    scores: dict[str, float]
    winner_margin: float
    dominant_inputs: list[str]
    probe_room: float
    update_bias: float

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "score": round(self.score, 4),
            "scores": {str(key): round(float(value), 4) for key, value in self.scores.items()},
            "winner_margin": round(self.winner_margin, 4),
            "dominant_inputs": list(self.dominant_inputs),
            "probe_room": round(self.probe_room, 4),
            "update_bias": round(self.update_bias, 4),
        }


def derive_learning_mode_state(
    *,
    self_state: Mapping[str, Any],
    body_recovery_guard: Mapping[str, Any],
    body_homeostasis_state: Mapping[str, Any],
    homeostasis_budget_state: Mapping[str, Any],
    protection_mode: Mapping[str, Any],
    initiative_readiness: Mapping[str, Any],
    agenda_state: Mapping[str, Any],
    agenda_window_state: Mapping[str, Any],
    commitment_state: Mapping[str, Any],
    attention_regulation_state: Mapping[str, Any],
    grice_guard_state: Mapping[str, Any],
    relational_continuity_state: Mapping[str, Any],
    social_topology_state: Mapping[str, Any],
    insight_event: Mapping[str, Any],
    identity_arc_kind: str = "",
    identity_arc_phase: str = "",
) -> LearningModeState:
    state_payload = dict(self_state or {})
    body_guard = dict(body_recovery_guard or {})
    body_homeostasis = dict(body_homeostasis_state or {})
    budget = dict(homeostasis_budget_state or {})
    protection = dict(protection_mode or {})
    initiative = dict(initiative_readiness or {})
    agenda = dict(agenda_state or {})
    agenda_window = dict(agenda_window_state or {})
    commitment = dict(commitment_state or {})
    attention = dict(attention_regulation_state or {})
    grice = dict(grice_guard_state or {})
    continuity = dict(relational_continuity_state or {})
    topology = dict(social_topology_state or {})
    insight = dict(insight_event or {})

    guard_state = str(body_guard.get("state") or "open").strip() or "open"
    guard_score = _float01(body_guard.get("score"))
    body_state = str(body_homeostasis.get("state") or "steady").strip() or "steady"
    body_score = _float01(body_homeostasis.get("score"))
    budget_state = str(budget.get("state") or "steady").strip() or "steady"
    budget_score = _float01(budget.get("score"))
    protection_mode_name = str(protection.get("mode") or "").strip()
    protection_strength = _float01(protection.get("strength"))
    initiative_state = str(initiative.get("state") or "hold").strip() or "hold"
    initiative_score = _float01(initiative.get("score"))
    agenda_name = str(agenda.get("state") or "hold").strip() or "hold"
    agenda_score = _float01(agenda.get("score"))
    agenda_window_name = str(agenda_window.get("state") or "long_hold").strip() or "long_hold"
    agenda_window_deferral_budget = _float01(agenda_window.get("deferral_budget"))
    commitment_mode = str(commitment.get("state") or "waver").strip() or "waver"
    commitment_target = str(commitment.get("target") or "hold").strip() or "hold"
    commitment_score = _float01(commitment.get("score"))
    attention_state = str(attention.get("state") or "selective_hold").strip() or "selective_hold"
    grice_state = str(grice.get("state") or "advise_openly").strip() or "advise_openly"
    continuity_state = str(continuity.get("state") or "distant").strip() or "distant"
    continuity_score = _float01(continuity.get("score"))
    topology_state = str(topology.get("state") or "ambient").strip() or "ambient"
    topology_score = _float01(topology.get("score"))
    topology_visibility = _float01(topology.get("visibility_pressure"))
    topology_hierarchy = _float01(topology.get("hierarchy_pressure"))
    insight_triggered = bool(insight.get("triggered", False))
    insight_orient_bias = _float01(insight.get("orient_bias"))
    stress = _float01(state_payload.get("stress"))
    recovery_need = _float01(state_payload.get("recovery_need"))
    degraded = bool(state_payload.get("degraded", False))
    carry_focus = str(state_payload.get("learning_mode_focus") or "").strip()
    carry_bias = _float01(state_payload.get("learning_mode_carry_bias"))

    probe_room = _clamp01(
        0.18
        + initiative_score * 0.24
        + agenda_score * (0.14 if agenda_name in {"step_forward", "repair", "revisit"} else 0.06)
        + continuity_score * (0.12 if continuity_state in {"reopening", "co_regulating"} else 0.04)
        + (0.08 if commitment_target in {"step_forward", "repair"} and commitment_mode == "commit" else 0.0)
        - guard_score * 0.22
        - max(body_score, budget_score) * 0.18
        - protection_strength * (0.18 if protection_mode_name in {"contain", "stabilize", "shield"} else 0.06)
        - topology_visibility * 0.12
        - topology_hierarchy * 0.1
        - stress * 0.08
        - recovery_need * 0.08
    )
    update_bias = _clamp01(
        0.14
        + (0.16 if commitment_mode == "commit" else 0.0) * max(commitment_score, 0.5)
        + (0.12 if agenda_name in {"repair", "revisit", "step_forward"} else 0.0) * max(agenda_score, 0.5)
        + insight_orient_bias * 0.14
        + (0.1 if identity_arc_phase in {"integrating", "consolidating"} else 0.0)
        - max(body_score, budget_score) * 0.08
        - guard_score * 0.08
    )

    scores = {
        "observe_only": _clamp01(
            0.14
            + (0.18 if attention_state == "selective_hold" else 0.0)
            + (0.12 if grice_state in {"hold_obvious_advice", "attune_without_repeating"} else 0.0)
            + topology_visibility * 0.08
            + topology_hierarchy * 0.08
            + (0.08 if agenda_window_name in {"next_same_group_window", "next_same_culture_window"} else 0.0)
            - probe_room * 0.18
        ),
        "test_small": _clamp01(
            0.08
            + probe_room * 0.34
            + (0.14 if initiative_state == "ready" else 0.08 if initiative_state == "tentative" else 0.0)
            + (0.14 if agenda_name == "step_forward" else 0.06 if agenda_name == "revisit" else 0.0)
            + (0.12 if commitment_target == "step_forward" and commitment_mode == "commit" else 0.0)
            + (0.08 if identity_arc_kind == "growing_edge" else 0.0)
            - topology_visibility * 0.06
            - topology_hierarchy * 0.06
        ),
        "repair_probe": _clamp01(
            0.08
            + probe_room * 0.18
            + (0.18 if protection_mode_name == "repair" else 0.0)
            + (0.14 if commitment_target in {"repair", "bond_protect"} and commitment_mode == "commit" else 0.0)
            + (0.12 if continuity_state in {"reopening", "holding_thread"} else 0.0)
            + (0.12 if agenda_name == "repair" else 0.0)
            + (0.1 if identity_arc_kind in {"repairing_bond", "holding_thread"} else 0.0)
            - max(body_score, budget_score) * 0.08
        ),
        "hold_and_wait": _clamp01(
            0.08
            + guard_score * 0.24
            + max(body_score, budget_score) * 0.18
            + protection_strength * (0.18 if protection_mode_name in {"contain", "stabilize", "shield"} else 0.08)
            + (0.14 if agenda_window_name == "long_hold" else 0.08 if agenda_window_name == "next_private_window" else 0.0)
            + (0.12 if attention_state == "reflex_guard" else 0.08 if attention_state == "split_guarded" else 0.0)
            + (0.1 if degraded else 0.0)
        ),
        "integrate_and_commit": _clamp01(
            0.06
            + update_bias * 0.26
            + (0.2 if commitment_mode == "commit" else 0.0) * max(commitment_score, 0.5)
            + (0.14 if commitment_target in {"step_forward", "repair", "bond_protect"} else 0.0)
            + (0.12 if continuity_state == "co_regulating" else 0.0)
            + (0.1 if identity_arc_phase in {"integrating", "consolidating"} else 0.0)
            + (0.08 if agenda_name in {"repair", "step_forward"} else 0.0)
            - guard_score * 0.1
            - max(body_score, budget_score) * 0.08
        ),
    }

    if guard_state == "recovery_first" or body_state == "depleted" or budget_state == "depleted":
        scores["hold_and_wait"] = _clamp01(scores["hold_and_wait"] + 0.22)
        scores["test_small"] *= 0.28
        scores["integrate_and_commit"] *= 0.36
    if agenda_window_name == "next_private_window":
        scores["hold_and_wait"] = _clamp01(scores["hold_and_wait"] + 0.08)
        scores["repair_probe"] = _clamp01(scores["repair_probe"] + 0.04)
        scores["test_small"] *= 0.82
    elif agenda_window_name in {"next_same_group_window", "next_same_culture_window"}:
        scores["observe_only"] = _clamp01(scores["observe_only"] + 0.08)
        scores["test_small"] *= 0.84
    elif agenda_window_name == "opportunistic_reentry":
        scores["test_small"] = _clamp01(scores["test_small"] + 0.06)
        scores["observe_only"] = _clamp01(scores["observe_only"] + 0.04)

    if insight_triggered and protection_mode_name not in {"contain", "stabilize", "shield"}:
        scores["observe_only"] = _clamp01(scores["observe_only"] + 0.06)
        scores["repair_probe"] = _clamp01(scores["repair_probe"] + insight_orient_bias * 0.08)

    if carry_focus in scores and carry_bias > 0.0:
        carry_scale = 0.14
        if carry_focus in {"observe_only", "hold_and_wait"}:
            carry_scale = 0.18
        elif carry_focus == "repair_probe":
            carry_scale = 0.16
        elif carry_focus == "integrate_and_commit":
            carry_scale = 0.1
        if carry_focus in {"test_small", "integrate_and_commit"} and agenda_window_name in {
            "next_private_window",
            "next_same_group_window",
            "next_same_culture_window",
            "long_hold",
        }:
            carry_scale *= 0.72
        if carry_focus in {"test_small", "integrate_and_commit"} and max(body_score, budget_score) >= 0.36:
            carry_scale *= 0.68
        scores[carry_focus] = _clamp01(scores[carry_focus] + carry_bias * carry_scale)

    state, winner_margin = _winner_and_margin(scores)
    dominant_inputs = _compact(
        [
            "body_recovery_guard" if guard_state in {"guarded", "recovery_first"} else "",
            "body_homeostasis_state" if body_state in {"recovering", "depleted"} else "",
            "homeostasis_budget_state" if budget_state in {"recovering", "depleted"} else "",
            "protection_mode" if protection_mode_name else "",
            "initiative_readiness" if initiative_state in {"tentative", "ready"} else "",
            "agenda_state" if agenda_name != "hold" else "",
            "agenda_window_state" if agenda_window_name != "now" else "",
            "commitment_state" if commitment_mode != "waver" else "",
            "attention_regulation_state" if attention_state != "selective_hold" else "",
            "grice_guard_state" if grice_state != "advise_openly" else "",
            "relational_continuity_state" if continuity_state != "distant" else "",
            "social_topology_state" if topology_state != "ambient" else "",
            "identity_arc" if identity_arc_kind else "",
            "insight_event" if insight_triggered else "",
            "overnight_learning_mode_carry" if carry_focus and carry_bias > 0.0 else "",
        ]
    )

    return LearningModeState(
        state=state,
        score=_clamp01(scores.get(state, 0.0)),
        scores=scores,
        winner_margin=winner_margin,
        dominant_inputs=dominant_inputs,
        probe_room=probe_room,
        update_bias=update_bias,
    )


def _winner_and_margin(scores: Mapping[str, float]) -> tuple[str, float]:
    items = sorted(
        ((str(key), _clamp01(float(value or 0.0))) for key, value in scores.items()),
        key=lambda item: (-item[1], item[0]),
    )
    if not items:
        return "observe_only", 0.0
    winner_name, winner_score = items[0]
    runner_up = items[1][1] if len(items) > 1 else 0.0
    return winner_name, round(_clamp01(winner_score - runner_up), 4)


def _compact(values: list[str]) -> list[str]:
    return [value for value in values if value]


def _float01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return _clamp01(numeric)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
