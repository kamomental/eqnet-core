from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


_ACUTE_RISK_FRAGMENTS = (
    "danger",
    "violence",
    "weapon",
    "sharp",
    "forced_entry",
    "intrusion",
    "attack",
)
_SOCIAL_RISK_FRAGMENTS = (
    "unsafe_person",
    "coercion",
    "threat",
    "escalation",
    "boundary_break",
    "harassment",
    "pursuit",
)
_HELP_RISK_FRAGMENTS = (
    "emergency",
    "injury",
    "collapse",
    "trapped",
    "panic",
)
_ROUTINE_TASK_PHASES = {"coordination", "co_work", "shared_task", "meal", "cooking", "setup", "cleanup"}
_PUBLIC_TOPOLOGIES = {"public_visible", "hierarchical", "threaded_group"}
_PRIVATE_PLACE_HINTS = {"home", "house", "apartment", "private_room", "bedroom", "living_space"}


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
        return "ordinary_context", 0.0
    winner = ordered[0][0]
    top = ordered[0][1]
    runner = ordered[1][1] if len(ordered) > 1 else 0.0
    return winner, max(0.0, min(1.0, top - runner))


def _fragment_signal(tokens: Sequence[str], fragments: Sequence[str]) -> float:
    normalized = [token.lower() for token in tokens if token]
    matches = 0
    for token in normalized:
        if any(fragment in token for fragment in fragments):
            matches += 1
    if not normalized or matches <= 0:
        return 0.0
    return _float01(matches / max(1.0, len(normalized) * 0.75))


def _context_affordance(
    *,
    scene_family: str,
    task_phase: str,
    public_exposure: float,
    private_enclosure: float,
    intrusion_signal: float,
) -> str:
    if intrusion_signal >= 0.42:
        return "shelter_breach"
    if scene_family == "repair_window":
        return "guarded_repair"
    if scene_family == "shared_world" or task_phase in _ROUTINE_TASK_PHASES:
        return "routine_task"
    if public_exposure >= 0.44:
        return "public_exposure"
    if private_enclosure >= 0.48:
        return "private_interior"
    return "open_context"


@dataclass(frozen=True)
class SituationRiskState:
    state: str
    score: float
    scores: dict[str, float]
    winner_margin: float
    context_affordance: str
    threat_signal: float
    acute_signal: float
    intrusion_signal: float
    immediacy: float
    intent_clarity: float
    escape_room: float
    relation_break: float
    deviation_from_expected: float
    dialogue_room: float
    dominant_inputs: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "score": round(self.score, 4),
            "scores": {str(key): round(float(value), 4) for key, value in self.scores.items()},
            "winner_margin": round(self.winner_margin, 4),
            "context_affordance": self.context_affordance,
            "threat_signal": round(self.threat_signal, 4),
            "acute_signal": round(self.acute_signal, 4),
            "intrusion_signal": round(self.intrusion_signal, 4),
            "immediacy": round(self.immediacy, 4),
            "intent_clarity": round(self.intent_clarity, 4),
            "escape_room": round(self.escape_room, 4),
            "relation_break": round(self.relation_break, 4),
            "deviation_from_expected": round(self.deviation_from_expected, 4),
            "dialogue_room": round(self.dialogue_room, 4),
            "dominant_inputs": list(self.dominant_inputs),
        }


def derive_situation_risk_state(
    *,
    current_risks: Sequence[str],
    scene_state: Mapping[str, Any] | None,
    self_state: Mapping[str, Any] | None,
) -> SituationRiskState:
    scene = dict(scene_state or {})
    state = dict(self_state or {})
    risk_tokens = [_text(item).lower() for item in current_risks if _text(item)]
    scene_family = _text(scene.get("scene_family")).lower()
    task_phase = _text(scene.get("task_phase")).lower()
    place_mode = _text(scene.get("place_mode")).lower()
    social_topology = _text(scene.get("social_topology")).lower()
    privacy_level = _float01(scene.get("privacy_level"))
    safety_margin = _float01(scene.get("safety_margin"))
    mobility_context = _text(scene.get("mobility_context")).lower()
    scene_tags = [_text(item).lower() for item in scene.get("scene_tags") or [] if _text(item)]

    public_exposure = _float01(
        0.46 * (1.0 if privacy_level <= 0.28 else 0.0)
        + 0.22 * (1.0 if social_topology in _PUBLIC_TOPOLOGIES else 0.0)
        + 0.14 * (1.0 if "socially_exposed" in scene_tags else 0.0)
        + 0.14 * (1.0 if mobility_context not in {"", "stationary"} else 0.0)
    )
    private_enclosure = _float01(
        0.42 * (1.0 if privacy_level >= 0.72 else 0.0)
        + 0.28 * (1.0 if place_mode in _PRIVATE_PLACE_HINTS else 0.0)
        + 0.14 * (1.0 if "private" in scene_tags else 0.0)
    )
    routine_context = _float01(
        0.34 * (1.0 if scene_family == "shared_world" else 0.0)
        + 0.24 * (1.0 if task_phase in _ROUTINE_TASK_PHASES else 0.0)
        + 0.18 * (1.0 if scene_family == "repair_window" else 0.0)
        + 0.16 * max(0.0, safety_margin - 0.4)
    )
    acute_signal_raw = _fragment_signal(risk_tokens, _ACUTE_RISK_FRAGMENTS)
    social_risk_signal = _fragment_signal(risk_tokens, _SOCIAL_RISK_FRAGMENTS)
    help_signal = _fragment_signal(risk_tokens, _HELP_RISK_FRAGMENTS)
    intrusion_signal = _fragment_signal(risk_tokens, ("forced_entry", "intrusion", "breach"))
    tool_visibility_only = bool(risk_tokens) and all(
        any(fragment in token for fragment in ("sharp", "tool"))
        for token in risk_tokens
    )
    routine_dampening = _float01(
        0.58 * routine_context if social_risk_signal <= 0.05 and intrusion_signal <= 0.05 else 0.0
    )
    if tool_visibility_only and routine_context >= 0.32:
        routine_dampening = max(routine_dampening, 0.84)
    acute_signal = _float01(acute_signal_raw * (1.0 - routine_dampening))
    threat_signal = _float01(max(acute_signal, social_risk_signal))

    trust_memory = _float01(state.get("trust_memory"))
    familiarity = _float01(state.get("familiarity"))
    attachment = _float01(state.get("attachment"))
    continuity_score = _float01(state.get("continuity_score"))
    trust_baseline = _float01(max(trust_memory, familiarity, attachment))

    relation_break = _float01(
        threat_signal
        * (
            0.5 * trust_baseline
            + 0.3 * continuity_score
            + 0.2 * private_enclosure
        )
    )
    deviation_from_expected = _float01(
        threat_signal
        * (
            0.36 * routine_context
            + 0.32 * trust_baseline
            + 0.16 * private_enclosure
            + 0.16 * continuity_score
        )
    )
    immediacy = _float01(
        0.48 * acute_signal
        + 0.16 * social_risk_signal
        + 0.14 * intrusion_signal
        + 0.12 * public_exposure
        + 0.1 * max(0.0, 0.5 - safety_margin)
    )
    intent_clarity = _float01(
        0.42 * acute_signal
        + 0.24 * social_risk_signal
        + 0.18 * intrusion_signal
        + 0.16 * relation_break
    )
    escape_room = _float01(
        0.42 * safety_margin
        + 0.18 * public_exposure
        + 0.14 * (1.0 if mobility_context not in {"", "stationary"} else 0.0)
        - 0.2 * intrusion_signal
        - 0.12 * private_enclosure
    )
    dialogue_room = _float01(
        0.54 * (1.0 - immediacy)
        + 0.22 * (1.0 - intent_clarity)
        + 0.16 * escape_room
        + 0.14 * routine_context
        - 0.18 * relation_break
    )
    context_affordance = _context_affordance(
        scene_family=scene_family,
        task_phase=task_phase,
        public_exposure=public_exposure,
        private_enclosure=private_enclosure,
        intrusion_signal=intrusion_signal,
    )

    scores = {
        "ordinary_context": _float01(
            0.54 * routine_context
            + 0.32 * safety_margin
            + 0.16 * dialogue_room
            - 0.52 * threat_signal
            - 0.24 * deviation_from_expected
        ),
        "guarded_context": _float01(
            0.28 * threat_signal
            + 0.24 * dialogue_room
            + 0.18 * public_exposure
            + 0.16 * private_enclosure
            + 0.14 * (1.0 - safety_margin)
            - 0.18 * immediacy
        ),
        "unstable_contact": _float01(
            0.34 * relation_break
            + 0.26 * deviation_from_expected
            + 0.18 * social_risk_signal
            + 0.16 * dialogue_room
            + 0.1 * trust_baseline
        ),
        "acute_threat": _float01(
            0.46 * immediacy
            + 0.26 * intent_clarity
            + 0.18 * (1.0 - escape_room)
            + 0.14 * public_exposure
        ),
        "emergency": _float01(
            0.5 * immediacy
            + 0.24 * (1.0 - escape_room)
            + 0.18 * intrusion_signal
            + 0.14 * help_signal
            + 0.12 * intent_clarity
        ),
    }
    winner, winner_margin = _winner_and_margin(scores)
    dominant_inputs = [
        label
        for label, enabled in (
            ("routine_context", routine_context >= 0.34),
            ("public_exposure", public_exposure >= 0.34),
            ("private_enclosure", private_enclosure >= 0.34),
            ("acute_risk", acute_signal >= 0.24),
            ("social_risk", social_risk_signal >= 0.24),
            ("intrusion_signal", intrusion_signal >= 0.24),
            ("relation_break", relation_break >= 0.24),
            ("deviation_from_expected", deviation_from_expected >= 0.24),
            ("low_escape_room", escape_room <= 0.32),
            ("low_dialogue_room", dialogue_room <= 0.32),
        )
        if enabled
    ]

    return SituationRiskState(
        state=winner,
        score=scores[winner],
        scores=scores,
        winner_margin=winner_margin,
        context_affordance=context_affordance,
        threat_signal=threat_signal,
        acute_signal=acute_signal,
        intrusion_signal=intrusion_signal,
        immediacy=immediacy,
        intent_clarity=intent_clarity,
        escape_room=escape_room,
        relation_break=relation_break,
        deviation_from_expected=deviation_from_expected,
        dialogue_room=dialogue_room,
        dominant_inputs=dominant_inputs,
    )
