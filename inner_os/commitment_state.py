from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class CommitmentState:
    state: str
    target: str
    accepted_cost: float
    score: float
    scores: dict[str, float]
    target_scores: dict[str, float]
    winner_margin: float
    dominant_inputs: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "target": self.target,
            "accepted_cost": round(self.accepted_cost, 4),
            "score": round(self.score, 4),
            "scores": {key: round(value, 4) for key, value in self.scores.items()},
            "target_scores": {key: round(value, 4) for key, value in self.target_scores.items()},
            "winner_margin": round(self.winner_margin, 4),
            "dominant_inputs": list(self.dominant_inputs),
        }


def derive_commitment_state(
    *,
    self_state: Mapping[str, Any],
    qualia_planner_view: Mapping[str, Any],
    terrain_readout: Mapping[str, Any],
    protection_mode: Mapping[str, Any],
    body_recovery_guard: Mapping[str, Any],
    initiative_readiness: Mapping[str, Any],
    initiative_followup_bias: Mapping[str, Any],
    temperament_estimate: Mapping[str, Any],
    memory_write_class: str,
    memory_write_class_reason: str,
    insight_event: Mapping[str, Any] | None = None,
) -> CommitmentState:
    self_payload = dict(self_state or {})
    qualia = dict(qualia_planner_view or {})
    terrain = dict(terrain_readout or {})
    protection = dict(protection_mode or {})
    body_guard = dict(body_recovery_guard or {})
    readiness = dict(initiative_readiness or {})
    followup = dict(initiative_followup_bias or {})
    temperament = dict(temperament_estimate or {})
    insight = dict(insight_event or {})

    stress = _clamp01(float(self_payload.get("stress", 0.0) or 0.0))
    recovery_need = _clamp01(float(self_payload.get("recovery_need", 0.0) or 0.0))
    body_load = _clamp01(float(qualia.get("body_load", 0.0) or 0.0))
    protection_bias = _clamp01(float(qualia.get("protection_bias", 0.0) or 0.0))
    degraded = 1.0 if bool(qualia.get("degraded", False)) else 0.0
    terrain_protect_bias = _clamp01(float(terrain.get("protect_bias", 0.0) or 0.0))
    terrain_approach_bias = _clamp01(float(terrain.get("approach_bias", 0.0) or 0.0))
    protection_mode_name = str(protection.get("mode") or "").strip()
    protection_strength = _clamp01(float(protection.get("strength", 0.0) or 0.0))
    protection_margin = _clamp01(float(protection.get("winner_margin", 0.0) or 0.0))
    guard_state = str(body_guard.get("state") or "open").strip() or "open"
    guard_score = _clamp01(float(body_guard.get("score", 0.0) or 0.0))
    readiness_state = str(readiness.get("state") or "hold").strip() or "hold"
    readiness_score = _clamp01(float(readiness.get("score", 0.0) or 0.0))
    readiness_margin = _clamp01(float(readiness.get("winner_margin", 0.0) or 0.0))
    followup_state = str(followup.get("state") or "hold").strip() or "hold"
    followup_score = _clamp01(float(followup.get("score", 0.0) or 0.0))
    leader_tendency = _clamp01(float(temperament.get("leader_tendency", 0.0) or 0.0))
    hero_tendency = _clamp01(float(temperament.get("hero_tendency", 0.0) or 0.0))
    bond_drive = _clamp01(float(temperament.get("bond_drive", 0.0) or 0.0))
    recovery_discipline = _clamp01(float(temperament.get("recovery_discipline", 0.0) or 0.0))
    protect_floor = _clamp01(float(temperament.get("protect_floor", 0.0) or 0.0))
    risk_tolerance = _clamp01(float(temperament.get("risk_tolerance", 0.0) or 0.0))
    insight_triggered = 1.0 if bool(insight.get("triggered", False)) else 0.0
    insight_orient_bias = _clamp01(float(insight.get("orient_bias", 0.0) or 0.0))
    timeline_coherence = _clamp01(float(self_payload.get("temporal_timeline_coherence", 0.0) or 0.0))
    reentry_pull = _clamp01(float(self_payload.get("temporal_reentry_pull", 0.0) or 0.0))
    supersession_pressure = _clamp01(float(self_payload.get("temporal_supersession_pressure", 0.0) or 0.0))
    continuity_pressure = _clamp01(float(self_payload.get("temporal_continuity_pressure", 0.0) or 0.0))
    relation_reentry_pull = _clamp01(float(self_payload.get("temporal_relation_reentry_pull", 0.0) or 0.0))
    if timeline_coherence <= 0.0:
        timeline_coherence = _clamp01(float(self_payload.get("temporal_timeline_bias", 0.0) or 0.0))
    if reentry_pull <= 0.0:
        reentry_pull = _clamp01(float(self_payload.get("temporal_reentry_bias", 0.0) or 0.0))
    if supersession_pressure <= 0.0:
        supersession_pressure = _clamp01(float(self_payload.get("temporal_supersession_bias", 0.0) or 0.0))
    if continuity_pressure <= 0.0:
        continuity_pressure = _clamp01(float(self_payload.get("temporal_continuity_bias", 0.0) or 0.0))
    if relation_reentry_pull <= 0.0:
        relation_reentry_pull = _clamp01(float(self_payload.get("temporal_relation_reentry_bias", 0.0) or 0.0))
    temporal_mode = str(self_payload.get("temporal_membrane_mode") or self_payload.get("temporal_membrane_focus") or "ambient").strip() or "ambient"

    accepted_cost = _clamp01(
        0.26 * body_load
        + 0.2 * stress
        + 0.18 * recovery_need
        + 0.16 * terrain_protect_bias
        + 0.12 * protection_bias
        + 0.08 * degraded
    )

    followup_open = followup_score if followup_state == "offer_next_step" else 0.0
    followup_reopen = followup_score if followup_state == "reopen_softly" else 0.0
    followup_hold = followup_score if followup_state == "hold" else 0.0
    repair_memory = 1.0 if memory_write_class == "repair_trace" else 0.0
    bond_memory = 1.0 if memory_write_class == "bond_protection" else 0.0
    safe_memory = 1.0 if memory_write_class == "safe_repeat" else 0.0

    target_scores = {
        "hold": _clamp01(
            0.3 * guard_score
            + 0.18 * protect_floor
            + 0.14 * followup_hold
            + 0.12 * degraded
            + 0.12 * terrain_protect_bias
            + 0.08 * (1.0 if protection_mode_name == "contain" else 0.0)
            + 0.12 * supersession_pressure
        ),
        "stabilize": _clamp01(
            0.28 * guard_score
            + 0.22 * protection_strength * (1.0 if protection_mode_name in {"stabilize", "shield"} else 0.0)
            + 0.16 * recovery_discipline
            + 0.1 * recovery_need
            + 0.08 * terrain_protect_bias
            + 0.06 * followup_reopen
            + 0.08 * supersession_pressure
            + 0.04 * timeline_coherence
        ),
        "repair": _clamp01(
            0.28 * protection_strength * (1.0 if protection_mode_name == "repair" else 0.0)
            + 0.16 * leader_tendency
            + 0.14 * followup_reopen
            + 0.1 * terrain_approach_bias
            + 0.1 * repair_memory
            + 0.06 * insight_triggered
            + 0.1 * reentry_pull
            + 0.1 * relation_reentry_pull
            + 0.04 * continuity_pressure
            - 0.08 * degraded
            - 0.05 * supersession_pressure
        ),
        "bond_protect": _clamp01(
            0.22 * bond_drive
            + 0.18 * bond_memory
            + 0.14 * leader_tendency
            + 0.12 * terrain_protect_bias
            + 0.1 * followup_reopen
            + 0.08 * protection_strength * (1.0 if protection_mode_name in {"contain", "repair"} else 0.0)
            + 0.08 * continuity_pressure
            + 0.06 * relation_reentry_pull
            - 0.06 * degraded
        ),
        "step_forward": _clamp01(
            0.26 * readiness_score * (1.0 if readiness_state == "ready" else 0.55)
            + 0.18 * terrain_approach_bias
            + 0.16 * followup_open
            + 0.14 * hero_tendency
            + 0.08 * risk_tolerance
            + 0.08 * safe_memory
            + 0.06 * reentry_pull
            + 0.04 * timeline_coherence
            - 0.14 * guard_score
            - 0.12 * terrain_protect_bias
            - 0.08 * degraded
            - 0.12 * supersession_pressure
        ),
    }
    if temporal_mode == "supersede":
        target_scores["hold"] = _clamp01(target_scores["hold"] + 0.04)
        target_scores["stabilize"] = _clamp01(target_scores["stabilize"] + 0.03)
        target_scores["step_forward"] = _clamp01(target_scores["step_forward"] - 0.04)
    elif temporal_mode == "reentry":
        target_scores["repair"] = _clamp01(target_scores["repair"] + 0.03)
        target_scores["step_forward"] = _clamp01(target_scores["step_forward"] + 0.02)
    target, target_margin = _winner_and_margin(target_scores)

    cross_pressure = min(guard_score, max(readiness_score, followup_open, followup_reopen))
    target_score = _clamp01(float(target_scores.get(target, 0.0) or 0.0))
    target_alignment = _target_alignment(
        target=target,
        target_score=target_score,
        guard_score=guard_score,
        readiness_score=readiness_score,
        followup_open=followup_open,
        followup_reopen=followup_reopen,
        protection_strength=protection_strength,
        leader_tendency=leader_tendency,
        hero_tendency=hero_tendency,
        bond_drive=bond_drive,
        recovery_discipline=recovery_discipline,
    )

    scores = {
        "waver": _clamp01(
            0.2 * (1.0 - target_margin)
            + 0.16 * cross_pressure
            + 0.12 * degraded
            + 0.12 * followup_hold
            + 0.08 * (1.0 - readiness_margin)
            + 0.08 * (1.0 - protection_margin)
            + 0.08 * supersession_pressure
        ),
        "settle": _clamp01(
            0.24 * target_score
            + 0.18 * target_margin
            + 0.16 * target_alignment
            + 0.1 * max(followup_reopen, followup_open)
            + 0.08 * max(readiness_margin, protection_margin)
            + 0.06 * insight_orient_bias
            + 0.04 * timeline_coherence
            - 0.08 * degraded
        ),
        "commit": _clamp01(
            0.28 * target_score
            + 0.24 * target_margin
            + 0.18 * target_alignment
            + 0.12 * accepted_cost
            + 0.08 * max(readiness_margin, protection_margin)
            + 0.08 * max(followup_open, followup_reopen)
            + 0.06 * reentry_pull
            + 0.04 * continuity_pressure
            - 0.08 * degraded
            - 0.05 * supersession_pressure
        ),
    }
    if (
        target == "step_forward"
        and readiness_state == "ready"
        and guard_state == "open"
        and target_score >= 0.26
        and target_margin >= 0.06
    ):
        scores["commit"] = _clamp01(scores["commit"] + 0.18)
        scores["settle"] = _clamp01(scores["settle"] + 0.06)
        scores["waver"] = _clamp01(scores["waver"] - 0.08)
    elif (
        target in {"repair", "bond_protect"}
        and guard_state != "recovery_first"
        and target_score >= 0.26
        and target_margin >= 0.05
    ):
        scores["commit"] = _clamp01(scores["commit"] + 0.16)
        scores["settle"] = _clamp01(scores["settle"] + 0.05)
        scores["waver"] = _clamp01(scores["waver"] - 0.07)
    elif (
        target in {"stabilize", "hold"}
        and target_score >= 0.24
        and (guard_state in {"guarded", "recovery_first"} or protection_mode_name in {"stabilize", "shield"})
    ):
        scores["commit"] = _clamp01(scores["commit"] + 0.12)
        scores["waver"] = _clamp01(scores["waver"] - 0.04)
    if guard_state == "recovery_first" and target in {"step_forward", "repair", "bond_protect"}:
        scores["commit"] = _clamp01(scores["commit"] * 0.45)
        scores["settle"] = _clamp01(scores["settle"] * 0.72)
        scores["waver"] = _clamp01(scores["waver"] + 0.12)
    if protection_mode_name in {"shield"} and target in {"step_forward", "repair"}:
        scores["commit"] = _clamp01(scores["commit"] * 0.38)
        scores["waver"] = _clamp01(scores["waver"] + 0.14)

    state, winner_margin = _winner_and_margin(scores)
    dominant_inputs = _compact(
        [
            "target_margin" if target_margin >= 0.14 else "",
            "accepted_cost" if accepted_cost >= 0.16 else "",
            "body_recovery_guard" if guard_state in {"guarded", "recovery_first"} else "",
            "initiative_readiness" if readiness_state in {"tentative", "ready"} and readiness_score >= 0.2 else "",
            "initiative_followup_bias" if max(followup_open, followup_reopen, followup_hold) >= 0.14 else "",
            "temperament_leader_trace" if target in {"repair", "bond_protect"} and leader_tendency >= 0.46 else "",
            "temperament_hero_trace" if target == "step_forward" and hero_tendency >= 0.46 else "",
            "temperament_guard_floor" if target in {"hold", "stabilize"} and protect_floor >= 0.5 else "",
            "repair_memory_trace" if repair_memory >= 1.0 else "",
            "bond_memory_trace" if bond_memory >= 1.0 else "",
            memory_write_class_reason if memory_write_class_reason else "",
            "insight_orientation" if insight_orient_bias >= 0.12 else "",
            "temporal_reentry_pull" if reentry_pull >= 0.22 and target in {"repair", "step_forward", "bond_protect"} else "",
            "temporal_supersession_pressure" if supersession_pressure >= 0.22 and target in {"hold", "stabilize"} else "",
            "temporal_timeline_coherence" if timeline_coherence >= 0.22 and state in {"settle", "commit"} else "",
        ]
    )

    return CommitmentState(
        state=state,
        target=target,
        accepted_cost=accepted_cost,
        score=_clamp01(float(scores.get(state, 0.0) or 0.0)),
        scores=scores,
        target_scores=target_scores,
        winner_margin=winner_margin,
        dominant_inputs=dominant_inputs,
    )


def _target_alignment(
    *,
    target: str,
    target_score: float,
    guard_score: float,
    readiness_score: float,
    followup_open: float,
    followup_reopen: float,
    protection_strength: float,
    leader_tendency: float,
    hero_tendency: float,
    bond_drive: float,
    recovery_discipline: float,
) -> float:
    if target == "step_forward":
        return _clamp01(0.38 * readiness_score + 0.22 * followup_open + 0.2 * hero_tendency + 0.12 * target_score)
    if target == "repair":
        return _clamp01(0.32 * protection_strength + 0.24 * followup_reopen + 0.18 * leader_tendency + 0.12 * target_score)
    if target == "bond_protect":
        return _clamp01(0.28 * bond_drive + 0.24 * leader_tendency + 0.16 * protection_strength + 0.12 * target_score)
    if target == "stabilize":
        return _clamp01(0.34 * guard_score + 0.22 * protection_strength + 0.18 * recovery_discipline + 0.12 * target_score)
    return _clamp01(0.4 * guard_score + 0.22 * recovery_discipline + 0.12 * target_score)


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
