from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Mapping, Sequence

from .affect_blend import AffectBlendState
from .conscious_workspace import ConsciousWorkspace
from .constraint_field import ConstraintField
from .interaction_option_search import InteractionOptionCandidate
from .scene_state import SceneState


@dataclass(frozen=True)
class EstimatedOtherPersonState:
    detail_room: float = 0.0
    detail_room_level: str = "medium"
    acknowledgement_need: float = 0.0
    acknowledgement_need_level: str = "medium"
    pressure_sensitivity: float = 0.0
    pressure_sensitivity_level: str = "medium"
    next_step_room: float = 0.0
    next_step_room_level: str = "medium"
    cues: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResonanceCandidateAssessment:
    family_id: str
    option_id: str
    base_weight: float = 0.0
    resonance_score: float = 0.0
    adjusted_weight: float = 0.0
    burden_risk: float = 0.0
    pressure_risk: float = 0.0
    felt_understanding: float = 0.0
    self_pacing_support: float = 0.0
    connection_preservation: float = 0.0
    continued_talk_support: float = 0.0
    rationale: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResonanceEvaluation:
    estimated_other_person_state: EstimatedOtherPersonState
    recommended_family_id: str = ""
    recommended_option_id: str = ""
    avoid_actions: tuple[str, ...] = ()
    prioritize_actions: tuple[str, ...] = ()
    expected_effects: tuple[str, ...] = ()
    assessments: tuple[ResonanceCandidateAssessment, ...] = ()
    cues: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["estimated_other_person_state"] = self.estimated_other_person_state.to_dict()
        payload["assessments"] = [assessment.to_dict() for assessment in self.assessments]
        return payload


OPTION_BEHAVIOR_VECTORS: dict[str, dict[str, float]] = {
    "attune": {
        "acknowledge_strength": 0.95,
        "probe_force": 0.14,
        "pace_support": 0.86,
        "connection_hold": 0.84,
        "forward_push": 0.22,
        "protective_hold": 0.34,
    },
    "wait": {
        "acknowledge_strength": 0.56,
        "probe_force": 0.04,
        "pace_support": 0.96,
        "connection_hold": 0.7,
        "forward_push": 0.04,
        "protective_hold": 0.56,
    },
    "repair": {
        "acknowledge_strength": 0.92,
        "probe_force": 0.18,
        "pace_support": 0.78,
        "connection_hold": 0.82,
        "forward_push": 0.18,
        "protective_hold": 0.46,
    },
    "co_move": {
        "acknowledge_strength": 0.34,
        "probe_force": 0.32,
        "pace_support": 0.34,
        "connection_hold": 0.66,
        "forward_push": 0.96,
        "protective_hold": 0.14,
    },
    "contain": {
        "acknowledge_strength": 0.6,
        "probe_force": 0.04,
        "pace_support": 0.78,
        "connection_hold": 0.56,
        "forward_push": 0.04,
        "protective_hold": 0.96,
    },
    "reflect": {
        "acknowledge_strength": 0.58,
        "probe_force": 0.12,
        "pace_support": 0.66,
        "connection_hold": 0.62,
        "forward_push": 0.16,
        "protective_hold": 0.32,
    },
    "clarify": {
        "acknowledge_strength": 0.32,
        "probe_force": 0.88,
        "pace_support": 0.18,
        "connection_hold": 0.48,
        "forward_push": 0.5,
        "protective_hold": 0.1,
    },
    "withdraw": {
        "acknowledge_strength": 0.18,
        "probe_force": 0.02,
        "pace_support": 0.52,
        "connection_hold": 0.12,
        "forward_push": 0.0,
        "protective_hold": 0.68,
    },
}


def evaluate_interaction_resonance(
    *,
    scene_state: SceneState,
    affect_blend: AffectBlendState,
    constraint_field: ConstraintField,
    conscious_workspace: ConsciousWorkspace,
    interaction_option_candidates: Sequence[InteractionOptionCandidate],
    current_risks: Sequence[str] = (),
) -> ResonanceEvaluation:
    estimated_other_person_state = _estimate_other_person_state(
        scene_state=scene_state,
        affect_blend=affect_blend,
        constraint_field=constraint_field,
        conscious_workspace=conscious_workspace,
        current_risks=current_risks,
    )
    raw_assessments: list[dict[str, Any]] = []
    for candidate in interaction_option_candidates:
        raw_assessments.append(
            _score_candidate(
                candidate=candidate,
                estimated_other_person_state=estimated_other_person_state,
                scene_state=scene_state,
                affect_blend=affect_blend,
                constraint_field=constraint_field,
                conscious_workspace=conscious_workspace,
            )
        )

    total_adjusted = sum(float(item["adjusted_weight"]) for item in raw_assessments) or 1.0
    assessments = tuple(
        sorted(
            (
                ResonanceCandidateAssessment(
                    family_id=str(item["family_id"]),
                    option_id=str(item["option_id"]),
                    base_weight=round(_clamp01(item["base_weight"]), 6),
                    resonance_score=round(_clamp01(item["resonance_score"]), 6),
                    adjusted_weight=round(_clamp01(item["adjusted_weight"] / total_adjusted), 6),
                    burden_risk=round(_clamp01(item["burden_risk"]), 6),
                    pressure_risk=round(_clamp01(item["pressure_risk"]), 6),
                    felt_understanding=round(_clamp01(item["felt_understanding"]), 6),
                    self_pacing_support=round(_clamp01(item["self_pacing_support"]), 6),
                    connection_preservation=round(_clamp01(item["connection_preservation"]), 6),
                    continued_talk_support=round(_clamp01(item["continued_talk_support"]), 6),
                    rationale=tuple(item["rationale"]),
                )
                for item in raw_assessments
            ),
            key=lambda item: item.adjusted_weight,
            reverse=True,
        )
    )
    recommended_family_id = assessments[0].family_id if assessments else ""
    recommended_option_id = assessments[0].option_id if assessments else ""
    avoid_actions = _derive_avoid_actions(
        estimated_other_person_state=estimated_other_person_state,
        constraint_field=constraint_field,
        top_assessment=assessments[0] if assessments else None,
    )
    prioritize_actions = _derive_prioritize_actions(
        estimated_other_person_state=estimated_other_person_state,
        constraint_field=constraint_field,
        top_assessment=assessments[0] if assessments else None,
    )
    expected_effects = _derive_expected_effects(assessments[0] if assessments else None)
    cues = _derive_resonance_cues(
        estimated_other_person_state=estimated_other_person_state,
        top_assessment=assessments[0] if assessments else None,
    )
    return ResonanceEvaluation(
        estimated_other_person_state=estimated_other_person_state,
        recommended_family_id=recommended_family_id,
        recommended_option_id=recommended_option_id,
        avoid_actions=avoid_actions,
        prioritize_actions=prioritize_actions,
        expected_effects=expected_effects,
        assessments=assessments,
        cues=cues,
    )


def rerank_interaction_option_candidates(
    *,
    interaction_option_candidates: Sequence[InteractionOptionCandidate],
    resonance_evaluation: ResonanceEvaluation | Mapping[str, Any] | None,
) -> list[InteractionOptionCandidate]:
    if not interaction_option_candidates:
        return []
    evaluation = resonance_evaluation
    if isinstance(evaluation, Mapping):
        assessment_map = {
            str(item.get("option_id") or ""): float(item.get("adjusted_weight", 0.0) or 0.0)
            for item in evaluation.get("assessments") or []
            if isinstance(item, Mapping)
        }
    elif isinstance(evaluation, ResonanceEvaluation):
        assessment_map = {
            assessment.option_id: assessment.adjusted_weight for assessment in evaluation.assessments
        }
    else:
        assessment_map = {}

    ranked = sorted(
        interaction_option_candidates,
        key=lambda candidate: assessment_map.get(candidate.option_id, candidate.relative_weight),
        reverse=True,
    )
    return [
        replace(
            candidate,
            relative_weight=round(
                _clamp01(assessment_map.get(candidate.option_id, candidate.relative_weight)),
                6,
            ),
        )
        for candidate in ranked
    ]


def _estimate_other_person_state(
    *,
    scene_state: SceneState,
    affect_blend: AffectBlendState,
    constraint_field: ConstraintField,
    conscious_workspace: ConsciousWorkspace,
    current_risks: Sequence[str],
) -> EstimatedOtherPersonState:
    reportability_limit = str(constraint_field.reportability_limit or "open")
    workspace_stability = _clamp01(conscious_workspace.workspace_stability)
    detail_room = _clamp01(
        0.88
        - constraint_field.body_cost * 0.32
        - constraint_field.boundary_pressure * 0.24
        - constraint_field.protective_bias * 0.18
        - affect_blend.distress * 0.12
        - (0.18 if reportability_limit == "withhold" else 0.08 if reportability_limit == "narrow" else 0.0)
        - (0.08 if "danger" in current_risks else 0.0)
        + workspace_stability * 0.08
    )
    acknowledgement_need = _clamp01(
        affect_blend.distress * 0.34
        + affect_blend.care * 0.22
        + constraint_field.repair_pressure * 0.18
        + affect_blend.conflict_level * 0.12
        + (0.08 if conscious_workspace.withheld_slice else 0.0)
        + (0.06 if conscious_workspace.reportable_slice else 0.0)
    )
    pressure_sensitivity = _clamp01(
        constraint_field.boundary_pressure * 0.34
        + constraint_field.protective_bias * 0.24
        + constraint_field.body_cost * 0.18
        + scene_state.norm_pressure * 0.12
        + affect_blend.distress * 0.08
        + (0.1 if reportability_limit == "withhold" else 0.04 if reportability_limit == "narrow" else 0.0)
        + (0.08 if "danger" in current_risks else 0.0)
    )
    next_step_room = _clamp01(
        constraint_field.shared_world_pressure * 0.42
        + workspace_stability * 0.18
        + detail_room * 0.16
        + (0.16 if scene_state.scene_family == "shared_world" else 0.0)
        - pressure_sensitivity * 0.22
        - constraint_field.body_cost * 0.1
    )
    cues: list[str] = []
    if detail_room <= 0.4:
        cues.append("listener_detail_room_low")
    if acknowledgement_need >= 0.55:
        cues.append("listener_acknowledgement_high")
    if pressure_sensitivity >= 0.55:
        cues.append("listener_pressure_sensitive")
    if next_step_room >= 0.58:
        cues.append("listener_next_step_room_open")
    return EstimatedOtherPersonState(
        detail_room=round(detail_room, 6),
        detail_room_level=_level(detail_room),
        acknowledgement_need=round(acknowledgement_need, 6),
        acknowledgement_need_level=_level(acknowledgement_need),
        pressure_sensitivity=round(pressure_sensitivity, 6),
        pressure_sensitivity_level=_level(pressure_sensitivity),
        next_step_room=round(next_step_room, 6),
        next_step_room_level=_level(next_step_room),
        cues=tuple(cues),
    )


def _score_candidate(
    *,
    candidate: InteractionOptionCandidate,
    estimated_other_person_state: EstimatedOtherPersonState,
    scene_state: SceneState,
    affect_blend: AffectBlendState,
    constraint_field: ConstraintField,
    conscious_workspace: ConsciousWorkspace,
) -> dict[str, Any]:
    behavior = dict(OPTION_BEHAVIOR_VECTORS.get(candidate.family_id, OPTION_BEHAVIOR_VECTORS["attune"]))
    behavior = _apply_candidate_modifiers(behavior=behavior, candidate=candidate)
    burden_risk = _clamp01(
        constraint_field.body_cost * 0.38
        + constraint_field.protective_bias * 0.18
        + behavior["probe_force"] * 0.18
        + behavior["forward_push"] * max(0.0, 0.55 - estimated_other_person_state.next_step_room) * 0.22
        + (0.08 if candidate.disclosure_depth == "medium" and estimated_other_person_state.detail_room < 0.55 else 0.0)
        - behavior["protective_hold"] * 0.12
        - behavior["pace_support"] * 0.08
    )
    pressure_risk = _clamp01(
        estimated_other_person_state.pressure_sensitivity * 0.42
        + behavior["probe_force"] * 0.28
        + behavior["forward_push"] * max(0.0, 0.65 - estimated_other_person_state.next_step_room) * 0.2
        - behavior["pace_support"] * 0.16
        - behavior["protective_hold"] * 0.1
    )
    felt_understanding = _clamp01(
        estimated_other_person_state.acknowledgement_need * 0.46
        + behavior["acknowledge_strength"] * 0.28
        + behavior["connection_hold"] * 0.14
        - pressure_risk * 0.16
    )
    self_pacing_support = _clamp01(
        max(0.0, 1.0 - estimated_other_person_state.detail_room) * 0.2
        + estimated_other_person_state.pressure_sensitivity * 0.18
        + behavior["pace_support"] * 0.28
        + behavior["protective_hold"] * 0.1
        - behavior["probe_force"] * 0.18
    )
    connection_preservation = _clamp01(
        behavior["connection_hold"] * 0.34
        + behavior["acknowledge_strength"] * 0.12
        + max(0.0, 0.6 - estimated_other_person_state.pressure_sensitivity) * 0.08
        + affect_blend.care * 0.12
        - (0.18 if candidate.family_id == "withdraw" and estimated_other_person_state.acknowledgement_need >= 0.55 else 0.0)
    )
    continued_talk_support = _clamp01(
        estimated_other_person_state.detail_room * 0.16
        + estimated_other_person_state.next_step_room * 0.18
        + felt_understanding * 0.18
        + self_pacing_support * 0.14
        + (behavior["forward_push"] * 0.12 if estimated_other_person_state.next_step_room >= 0.55 else 0.0)
        - pressure_risk * 0.22
        - burden_risk * 0.14
    )
    fit_bonus, fit_rationales = _family_fit_bonus(
        candidate=candidate,
        estimated_other_person_state=estimated_other_person_state,
        constraint_field=constraint_field,
        conscious_workspace=conscious_workspace,
        scene_state=scene_state,
    )
    resonance_score = _clamp01(
        felt_understanding * 0.28
        + self_pacing_support * 0.24
        + connection_preservation * 0.2
        + continued_talk_support * 0.16
        - burden_risk * 0.16
        - pressure_risk * 0.18
        + fit_bonus
    )
    adjusted_weight = _clamp01(candidate.relative_weight * 0.48 + resonance_score * 0.52)
    rationale = list(fit_rationales)
    if burden_risk >= 0.52:
        rationale.append("burden_risk_high")
    if pressure_risk >= 0.52:
        rationale.append("pressure_risk_high")
    if felt_understanding >= 0.58:
        rationale.append("felt_understanding_high")
    if self_pacing_support >= 0.58:
        rationale.append("self_pacing_support_high")
    if continued_talk_support >= 0.52:
        rationale.append("continued_talk_support_high")
    return {
        "family_id": candidate.family_id,
        "option_id": candidate.option_id,
        "base_weight": candidate.relative_weight,
        "resonance_score": resonance_score,
        "adjusted_weight": adjusted_weight,
        "burden_risk": burden_risk,
        "pressure_risk": pressure_risk,
        "felt_understanding": felt_understanding,
        "self_pacing_support": self_pacing_support,
        "connection_preservation": connection_preservation,
        "continued_talk_support": continued_talk_support,
        "rationale": tuple(rationale),
    }


def _apply_candidate_modifiers(
    *,
    behavior: Mapping[str, float],
    candidate: InteractionOptionCandidate,
) -> dict[str, float]:
    adjusted = dict(behavior)
    if candidate.disclosure_depth == "minimal":
        adjusted["probe_force"] = _clamp01(adjusted["probe_force"] - 0.08)
        adjusted["pace_support"] = _clamp01(adjusted["pace_support"] + 0.04)
        adjusted["protective_hold"] = _clamp01(adjusted["protective_hold"] + 0.04)
    elif candidate.disclosure_depth == "medium":
        adjusted["probe_force"] = _clamp01(adjusted["probe_force"] + 0.06)
        adjusted["forward_push"] = _clamp01(adjusted["forward_push"] + 0.04)

    if "defer" in candidate.timing_mode or "slow" in candidate.timing_mode or "held" in candidate.timing_mode:
        adjusted["pace_support"] = _clamp01(adjusted["pace_support"] + 0.06)
        adjusted["probe_force"] = _clamp01(adjusted["probe_force"] - 0.04)
    if "step" in candidate.timing_mode or "forward" in candidate.timing_mode:
        adjusted["forward_push"] = _clamp01(adjusted["forward_push"] + 0.06)
    if candidate.boundary_mode in {"respectful", "protective", "guarded", "softened"}:
        adjusted["protective_hold"] = _clamp01(adjusted["protective_hold"] + 0.08)
        adjusted["probe_force"] = _clamp01(adjusted["probe_force"] - 0.04)
    return adjusted


def _family_fit_bonus(
    *,
    candidate: InteractionOptionCandidate,
    estimated_other_person_state: EstimatedOtherPersonState,
    constraint_field: ConstraintField,
    conscious_workspace: ConsciousWorkspace,
    scene_state: SceneState,
) -> tuple[float, tuple[str, ...]]:
    family_id = candidate.family_id
    bonus = 0.0
    rationale: list[str] = []
    if estimated_other_person_state.acknowledgement_need >= 0.55 and family_id in {"attune", "repair"}:
        bonus += 0.08
        rationale.append("fit_acknowledgement")
    if estimated_other_person_state.pressure_sensitivity >= 0.55 and family_id in {"wait", "contain"}:
        bonus += 0.08
        rationale.append("fit_low_pressure")
    if estimated_other_person_state.next_step_room >= 0.58 and family_id == "co_move":
        bonus += 0.08
        rationale.append("fit_next_step_room")
    if scene_state.scene_family == "shared_world" and family_id == "co_move":
        bonus += 0.12
        rationale.append("fit_shared_world_scene")
    if constraint_field.shared_world_pressure >= 0.58 and family_id == "co_move":
        bonus += 0.08
        rationale.append("fit_shared_world_pressure")
    if estimated_other_person_state.detail_room >= 0.58 and family_id == "clarify":
        bonus += 0.06
        rationale.append("fit_detail_room")
    if constraint_field.repair_pressure >= 0.48 and family_id == "repair":
        bonus += 0.08
        rationale.append("fit_repair_pressure")
    if scene_state.scene_family == "reverent_distance" and family_id == "wait":
        bonus += 0.06
        rationale.append("fit_scene_distance")
    if conscious_workspace.workspace_mode == "guarded_foreground" and family_id in {"clarify", "co_move"}:
        bonus -= 0.12
        rationale.append("mismatch_guarded_workspace")
    if candidate.disclosure_depth == "medium" and estimated_other_person_state.detail_room <= 0.4:
        bonus -= 0.08
        rationale.append("mismatch_detail_room")
    if scene_state.scene_family == "shared_world" and family_id == "attune":
        bonus -= 0.04
        rationale.append("mismatch_shared_world_attune")
    if family_id == "withdraw" and estimated_other_person_state.acknowledgement_need >= 0.45:
        bonus -= 0.08
        rationale.append("mismatch_acknowledgement_need")
    return bonus, tuple(rationale)


def _derive_avoid_actions(
    *,
    estimated_other_person_state: EstimatedOtherPersonState,
    constraint_field: ConstraintField,
    top_assessment: ResonanceCandidateAssessment | None,
) -> tuple[str, ...]:
    avoid: list[str] = []
    if estimated_other_person_state.pressure_sensitivity >= 0.45:
        avoid.extend(["press_for_detail", "stack_questions"])
    if estimated_other_person_state.detail_room <= 0.4:
        avoid.append("rush_to_solution")
    if estimated_other_person_state.acknowledgement_need >= 0.48:
        avoid.append("skip_acknowledgement")
    if constraint_field.repair_pressure >= 0.48:
        avoid.append("ignore_repair_signal")
    if top_assessment and top_assessment.pressure_risk >= 0.5:
        avoid.append("move_too_fast")
    return tuple(dict.fromkeys(avoid))


def _derive_prioritize_actions(
    *,
    estimated_other_person_state: EstimatedOtherPersonState,
    constraint_field: ConstraintField,
    top_assessment: ResonanceCandidateAssessment | None,
) -> tuple[str, ...]:
    prioritize: list[str] = []
    if estimated_other_person_state.acknowledgement_need >= 0.45:
        prioritize.append("acknowledge_current_state")
    if estimated_other_person_state.pressure_sensitivity >= 0.45 or estimated_other_person_state.detail_room <= 0.45:
        prioritize.append("leave_talking_room")
    if constraint_field.repair_pressure >= 0.45:
        prioritize.append("reduce_force")
    if estimated_other_person_state.next_step_room >= 0.58:
        prioritize.append("offer_small_next_step")
    else:
        prioritize.append("keep_return_point")
    if top_assessment and top_assessment.self_pacing_support >= 0.55:
        prioritize.append("support_self_pacing")
    return tuple(dict.fromkeys(prioritize))


def _derive_expected_effects(
    top_assessment: ResonanceCandidateAssessment | None,
) -> tuple[str, ...]:
    if top_assessment is None:
        return ()
    effects: list[str] = []
    if top_assessment.pressure_risk <= 0.42:
        effects.append("reduce_immediate_pressure")
    if top_assessment.felt_understanding >= 0.52:
        effects.append("help_other_feel_received")
    if top_assessment.self_pacing_support >= 0.52:
        effects.append("let_other_choose_talk_pace")
    if top_assessment.connection_preservation >= 0.5:
        effects.append("keep_connection_open")
    if top_assessment.continued_talk_support >= 0.38 or top_assessment.family_id == "co_move":
        effects.append("keep_next_turn_open")
    return tuple(dict.fromkeys(effects))


def _derive_resonance_cues(
    *,
    estimated_other_person_state: EstimatedOtherPersonState,
    top_assessment: ResonanceCandidateAssessment | None,
) -> tuple[str, ...]:
    cues = list(estimated_other_person_state.cues)
    if top_assessment is not None:
        cues.append(f"resonance_top:{top_assessment.family_id}")
        if top_assessment.pressure_risk >= 0.52:
            cues.append("resonance_pressure_high")
        if top_assessment.felt_understanding >= 0.58:
            cues.append("resonance_felt_understanding")
    return tuple(dict.fromkeys(cues))


def _level(value: float) -> str:
    if value >= 0.67:
        return "high"
    if value <= 0.33:
        return "low"
    return "medium"


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))
