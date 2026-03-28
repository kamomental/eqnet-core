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


def _winner_and_margin(scores: Mapping[str, float]) -> tuple[str, float]:
    ordered = sorted(
        ((str(key), _float01(value)) for key, value in dict(scores).items()),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )
    if not ordered:
        return "observe", 0.0
    winner = ordered[0][0]
    top = ordered[0][1]
    runner = ordered[1][1] if len(ordered) > 1 else 0.0
    return winner, max(0.0, min(1.0, top - runner))


@dataclass(frozen=True)
class EmergencyPosture:
    state: str
    score: float
    scores: dict[str, float]
    winner_margin: float
    dialogue_permission: str
    primary_action: str
    supportive_actions: list[str]
    dominant_inputs: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "score": round(self.score, 4),
            "scores": {str(key): round(float(value), 4) for key, value in self.scores.items()},
            "winner_margin": round(self.winner_margin, 4),
            "dialogue_permission": self.dialogue_permission,
            "primary_action": self.primary_action,
            "supportive_actions": list(self.supportive_actions),
            "dominant_inputs": list(self.dominant_inputs),
        }


def derive_emergency_posture(
    *,
    situation_risk_state: Mapping[str, Any] | None,
    constraint_field: Mapping[str, Any] | None,
    protection_mode: Mapping[str, Any] | None,
    body_recovery_guard: Mapping[str, Any] | None,
    body_homeostasis_state: Mapping[str, Any] | None,
    homeostasis_budget_state: Mapping[str, Any] | None,
) -> EmergencyPosture:
    risk = dict(situation_risk_state or {})
    constraint = dict(constraint_field or {})
    protection = dict(protection_mode or {})
    body_guard = dict(body_recovery_guard or {})
    body_homeostasis = dict(body_homeostasis_state or {})
    homeostasis_budget = dict(homeostasis_budget_state or {})

    risk_state = _text(risk.get("state"))
    context_affordance = _text(risk.get("context_affordance"))
    acute_signal = _float01(risk.get("acute_signal"))
    intrusion_signal = _float01(risk.get("intrusion_signal"))
    immediacy = _float01(risk.get("immediacy"))
    intent_clarity = _float01(risk.get("intent_clarity"))
    escape_room = _float01(risk.get("escape_room"))
    relation_break = _float01(risk.get("relation_break"))
    dialogue_room = _float01(risk.get("dialogue_room"))
    risk_emergency = _float01((risk.get("scores") or {}).get("emergency"))
    risk_acute = _float01((risk.get("scores") or {}).get("acute_threat"))
    risk_unstable = _float01((risk.get("scores") or {}).get("unstable_contact"))
    risk_guarded = _float01((risk.get("scores") or {}).get("guarded_context"))
    risk_ordinary = _float01((risk.get("scores") or {}).get("ordinary_context"))

    protection_mode_name = _text(protection.get("mode"))
    protection_strength = _float01(protection.get("strength"))
    guard_state = _text(body_guard.get("state"))
    guard_score = _float01(body_guard.get("score"))
    body_state = _text(body_homeostasis.get("state"))
    body_score = _float01(body_homeostasis.get("score"))
    budget_state = _text(homeostasis_budget.get("state"))
    budget_score = _float01(homeostasis_budget.get("score"))
    boundary_pressure = _float01(constraint.get("boundary_pressure"))
    protective_bias = _float01(constraint.get("protective_bias"))

    protection_pressure = _float01(
        max(
            protection_strength if protection_mode_name in {"contain", "stabilize", "shield"} else 0.0,
            guard_score if guard_state in {"guarded", "recovery_first"} else 0.0,
            body_score if body_state in {"depleted", "recovering"} else 0.0,
            budget_score if budget_state == "depleted" else 0.0,
            protective_bias,
            boundary_pressure,
        )
    )

    scores = {
        "observe": _float01(
            0.48 * risk_ordinary
            + 0.24 * dialogue_room
            + 0.14 * escape_room
            - 0.18 * immediacy
        ),
        "de_escalate": _float01(
            0.36 * risk_unstable
            + 0.18 * risk_guarded
            + 0.2 * dialogue_room
            + 0.16 * boundary_pressure
            + 0.14 * protection_pressure
        ),
        "create_distance": _float01(
            0.36 * risk_acute
            + 0.16 * risk_guarded
            + 0.22 * max(0.0, 1.0 - dialogue_room)
            + 0.16 * protection_pressure
            + 0.1 * escape_room
        ),
        "exit": _float01(
            0.38 * risk_emergency
            + 0.24 * risk_acute
            + 0.16 * escape_room
            + 0.12 * intrusion_signal
            + 0.12 * protection_pressure
        ),
        "seek_help": _float01(
            0.34 * risk_emergency
            + 0.22 * max(0.0, 0.45 - escape_room)
            + 0.16 * intrusion_signal
            + 0.16 * protection_pressure
            + 0.12 * acute_signal
        ),
        "emergency_protect": _float01(
            0.38 * risk_emergency
            + 0.2 * risk_acute
            + 0.18 * protection_pressure
            + 0.14 * max(0.0, 0.4 - escape_room)
            + 0.12 * intent_clarity
        ),
    }

    winner, winner_margin = _winner_and_margin(scores)
    dialogue_permission = {
        "observe": "allow_short",
        "de_escalate": "boundary_only",
        "create_distance": "boundary_only",
        "exit": "avoid_dialogue",
        "seek_help": "avoid_dialogue",
        "emergency_protect": "avoid_dialogue",
    }.get(winner, "allow_short")
    primary_action = {
        "observe": "observe_context_shift",
        "de_escalate": "set_clear_boundary",
        "create_distance": "create_distance",
        "exit": "exit_space",
        "seek_help": "seek_help",
        "emergency_protect": "protect_immediately",
    }.get(winner, "observe_context_shift")
    supportive_actions = {
        "observe": ["assess_context_shift", "keep_words_short"],
        "de_escalate": ["lower_heat", "keep_words_short", "watch_reaction"],
        "create_distance": ["orient_to_exit", "keep_words_short", "avoid_negotiation"],
        "exit": ["move_to_safety", "terminate_contact", "seek_help_if_needed"],
        "seek_help": ["make_risk_visible", "terminate_contact", "move_to_support"],
        "emergency_protect": ["protect_others_if_present", "reduce_exposure", "terminate_contact"],
    }.get(winner, ["assess_context_shift"])
    dominant_inputs = [
        label
        for label, enabled in (
            ("acute_threat", risk_acute >= 0.34 or risk_state == "acute_threat"),
            ("emergency_risk", risk_emergency >= 0.34 or risk_state == "emergency"),
            ("unstable_contact", risk_unstable >= 0.34 or risk_state == "unstable_contact"),
            ("low_dialogue_room", dialogue_room <= 0.34),
            ("low_escape_room", escape_room <= 0.34),
            ("intrusion_signal", intrusion_signal >= 0.24 or context_affordance == "shelter_breach"),
            ("relation_break", relation_break >= 0.24),
            ("protection_pressure", protection_pressure >= 0.34),
        )
        if enabled
    ]

    return EmergencyPosture(
        state=winner,
        score=scores[winner],
        scores=scores,
        winner_margin=winner_margin,
        dialogue_permission=dialogue_permission,
        primary_action=primary_action,
        supportive_actions=supportive_actions,
        dominant_inputs=dominant_inputs,
    )
