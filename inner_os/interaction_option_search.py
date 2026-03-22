from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .interaction.models import LiveInteractionRegulation, RelationalMood, SituationState
from .scene_state import SceneState


@dataclass(frozen=True)
class ActionFamilySpec:
    family_id: str
    base_bias: float
    scene_affinity: tuple[tuple[str, float], ...]
    tag_affinity: tuple[tuple[str, float], ...]
    feature_weights: tuple[tuple[str, float], ...]
    preferred_disclosure: str
    preferred_timing: str
    preferred_boundary: str
    next_moves: tuple[str, ...]


@dataclass(frozen=True)
class ActionFamilyActivation:
    family_id: str
    raw_score: float
    relative_weight: float
    rationale: tuple[str, ...]


@dataclass(frozen=True)
class InteractionOptionCandidate:
    family_id: str
    option_id: str
    relative_weight: float
    disclosure_depth: str
    timing_mode: str
    boundary_mode: str
    next_moves: tuple[str, ...]
    rationale: tuple[str, ...]


DEFAULT_ACTION_FAMILY_LIBRARY: tuple[ActionFamilySpec, ...] = (
    ActionFamilySpec(
        family_id="attune",
        base_bias=0.18,
        scene_affinity=(
            ("attuned_presence", 0.24),
            ("co_present", 0.12),
            ("shared_world", 0.06),
        ),
        tag_affinity=(("private", 0.08),),
        feature_weights=(
            ("care_signal", 0.28),
            ("contact_readiness", 0.22),
            ("shared_attention", 0.12),
            ("norm_room", 0.06),
        ),
        preferred_disclosure="light",
        preferred_timing="soft_start",
        preferred_boundary="permeable",
        next_moves=("stay_visible", "invite_state", "gentle_approach"),
    ),
    ActionFamilySpec(
        family_id="wait",
        base_bias=0.12,
        scene_affinity=(
            ("reverent_distance", 0.28),
            ("guarded_boundary", 0.18),
        ),
        tag_affinity=(
            ("public", 0.18),
            ("high_norm", 0.18),
        ),
        feature_weights=(
            ("reverence_signal", 0.34),
            ("boundary_need", 0.24),
            ("norm_pressure", 0.16),
        ),
        preferred_disclosure="minimal",
        preferred_timing="deferred_return",
        preferred_boundary="respectful",
        next_moves=("hold_presence", "defer", "leave_return_point"),
    ),
    ActionFamilySpec(
        family_id="repair",
        base_bias=0.1,
        scene_affinity=(
            ("repair_window", 0.34),
            ("guarded_boundary", 0.12),
        ),
        tag_affinity=(("goal:repair", 0.24),),
        feature_weights=(
            ("repair_need", 0.42),
            ("care_signal", 0.16),
            ("contact_readiness", 0.08),
        ),
        preferred_disclosure="minimal",
        preferred_timing="slow_reentry",
        preferred_boundary="softened",
        next_moves=("name_overreach", "reduce_force", "reopen_carefully"),
    ),
    ActionFamilySpec(
        family_id="co_move",
        base_bias=0.08,
        scene_affinity=(
            ("shared_world", 0.34),
            ("attuned_presence", 0.08),
        ),
        tag_affinity=(
            ("task:coordination", 0.18),
            ("task:co_work", 0.18),
        ),
        feature_weights=(
            ("shared_world_signal", 0.34),
            ("future_signal", 0.24),
            ("contact_readiness", 0.12),
            ("privacy_room", 0.06),
        ),
        preferred_disclosure="medium",
        preferred_timing="step_forward",
        preferred_boundary="forward_open",
        next_moves=("synchronize", "map_next_step", "pace_match"),
    ),
    ActionFamilySpec(
        family_id="contain",
        base_bias=0.08,
        scene_affinity=(("guarded_boundary", 0.36),),
        tag_affinity=(
            ("risk:danger", 0.32),
            ("high_load", 0.18),
        ),
        feature_weights=(
            ("containment_need", 0.44),
            ("boundary_need", 0.22),
        ),
        preferred_disclosure="minimal",
        preferred_timing="stabilize_first",
        preferred_boundary="protective",
        next_moves=("secure_boundary", "stabilize", "reduce_force"),
    ),
    ActionFamilySpec(
        family_id="reflect",
        base_bias=0.08,
        scene_affinity=(
            ("attuned_presence", 0.18),
            ("reverent_distance", 0.08),
        ),
        tag_affinity=(("private", 0.08),),
        feature_weights=(
            ("reflection_need", 0.34),
            ("ambiguity_tolerance", 0.18),
            ("boundary_need", 0.08),
        ),
        preferred_disclosure="light",
        preferred_timing="open_pause",
        preferred_boundary="open_ambiguity",
        next_moves=("hold_meaning_open", "observe_more", "defer_closure"),
    ),
    ActionFamilySpec(
        family_id="clarify",
        base_bias=0.06,
        scene_affinity=(
            ("shared_world", 0.18),
            ("co_present", 0.06),
        ),
        tag_affinity=(("task:ongoing", 0.06),),
        feature_weights=(
            ("clarity_need", 0.34),
            ("contact_readiness", 0.08),
            ("shared_attention", 0.12),
        ),
        preferred_disclosure="light",
        preferred_timing="narrow_check",
        preferred_boundary="careful_precision",
        next_moves=("narrow_question", "check_visible", "confirm_scope"),
    ),
    ActionFamilySpec(
        family_id="withdraw",
        base_bias=0.04,
        scene_affinity=(
            ("guarded_boundary", 0.16),
            ("reverent_distance", 0.1),
        ),
        tag_affinity=(("low_safety", 0.24),),
        feature_weights=(
            ("withdraw_need", 0.42),
            ("boundary_need", 0.18),
        ),
        preferred_disclosure="minimal",
        preferred_timing="step_back",
        preferred_boundary="closed_boundary",
        next_moves=("step_back", "reduce_contact", "preserve_line"),
    ),
)


def compute_action_family_activations(
    *,
    scene_state: SceneState,
    situation_state: SituationState,
    relational_mood: RelationalMood,
    live_regulation: LiveInteractionRegulation,
    constraint_field: Mapping[str, Any] | None = None,
    family_library: Sequence[ActionFamilySpec] = DEFAULT_ACTION_FAMILY_LIBRARY,
) -> list[ActionFamilyActivation]:
    features = _derive_feature_map(
        scene_state=scene_state,
        situation_state=situation_state,
        relational_mood=relational_mood,
        live_regulation=live_regulation,
        constraint_field=constraint_field,
    )
    admissible_families = {
        str(item)
        for item in (constraint_field or {}).get("admissible_families") or []
        if str(item).strip()
    }
    raw_scores: list[tuple[ActionFamilySpec, float, tuple[str, ...]]] = []
    for spec in family_library:
        score = spec.base_bias
        rationale: list[str] = []
        for scene_name, weight in spec.scene_affinity:
            if scene_name == scene_state.scene_family:
                score += weight
                rationale.append(f"scene:{scene_name}")
        active_tags = set(scene_state.scene_tags)
        for tag_name, weight in spec.tag_affinity:
            if tag_name in active_tags:
                score += weight
                rationale.append(f"tag:{tag_name}")
        for feature_name, weight in spec.feature_weights:
            feature_value = features.get(feature_name, 0.0)
            score += feature_value * weight
            if feature_value >= 0.44:
                rationale.append(f"{feature_name}:{round(feature_value, 2)}")
        if admissible_families and spec.family_id not in admissible_families:
            score -= 0.45
            rationale.append("constraint:inadmissible")
        raw_scores.append((spec, score, tuple(rationale)))

    weights = _softmax(
        (score for _, score, _ in raw_scores),
        temperature=float((constraint_field or {}).get("option_temperature", 1.0) or 1.0),
    )
    activations = [
        ActionFamilyActivation(
            family_id=spec.family_id,
            raw_score=round(score, 6),
            relative_weight=round(weight, 6),
            rationale=rationale,
        )
        for (spec, score, rationale), weight in zip(raw_scores, weights)
    ]
    return sorted(activations, key=lambda item: item.relative_weight, reverse=True)


def generate_interaction_option_candidates(
    *,
    scene_state: SceneState,
    situation_state: SituationState,
    relational_mood: RelationalMood,
    live_regulation: LiveInteractionRegulation,
    constraint_field: Mapping[str, Any] | None = None,
    family_library: Sequence[ActionFamilySpec] = DEFAULT_ACTION_FAMILY_LIBRARY,
    min_candidates: int = 3,
    max_candidates: int = 8,
    coverage_target: float = 0.84,
) -> list[InteractionOptionCandidate]:
    activations = compute_action_family_activations(
        scene_state=scene_state,
        situation_state=situation_state,
        relational_mood=relational_mood,
        live_regulation=live_regulation,
        constraint_field=constraint_field,
        family_library=family_library,
    )
    spec_map = {spec.family_id: spec for spec in family_library}
    selected = _select_candidate_families(
        activations=activations,
        min_candidates=min_candidates,
        max_candidates=max_candidates,
        coverage_target=coverage_target,
    )
    options: list[InteractionOptionCandidate] = []
    for activation in selected:
        spec = spec_map[activation.family_id]
        options.append(
            InteractionOptionCandidate(
                family_id=spec.family_id,
                option_id=f"{spec.family_id}:{scene_state.scene_family}",
                relative_weight=activation.relative_weight,
                disclosure_depth=_resolve_disclosure_depth(
                    preferred=spec.preferred_disclosure,
                    scene_state=scene_state,
                    live_regulation=live_regulation,
                ),
                timing_mode=_resolve_timing_mode(
                    preferred=spec.preferred_timing,
                    scene_state=scene_state,
                    live_regulation=live_regulation,
                ),
                boundary_mode=_resolve_boundary_mode(
                    preferred=spec.preferred_boundary,
                    scene_state=scene_state,
                    live_regulation=live_regulation,
                ),
                next_moves=spec.next_moves,
                rationale=activation.rationale,
            )
        )
    return options


def _select_candidate_families(
    *,
    activations: Sequence[ActionFamilyActivation],
    min_candidates: int,
    max_candidates: int,
    coverage_target: float,
) -> list[ActionFamilyActivation]:
    selected: list[ActionFamilyActivation] = []
    cumulative = 0.0
    for activation in activations:
        if len(selected) >= max_candidates:
            break
        selected.append(activation)
        cumulative += activation.relative_weight
        if len(selected) >= min_candidates and cumulative >= coverage_target:
            break
    return selected


def _derive_feature_map(
    *,
    scene_state: SceneState,
    situation_state: SituationState,
    relational_mood: RelationalMood,
    live_regulation: LiveInteractionRegulation,
    constraint_field: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    shared_attention = _clamp01(
        situation_state.shared_attention * 0.7 + live_regulation.shared_attention_active * 0.3
    )
    repair_need = _clamp01(
        (1.0 if situation_state.repair_window_open or live_regulation.repair_window_open else 0.0) * 0.45
        + live_regulation.strained_pause * 0.45
        + max(0.0, 0.4 - shared_attention) * 0.25
    )
    contact_readiness = _clamp01(
        relational_mood.care * 0.24
        + relational_mood.confidence_signal * 0.24
        + shared_attention * 0.18
        + scene_state.safety_margin * 0.22
        - scene_state.environmental_load * 0.18
    )
    boundary_need = _clamp01(
        scene_state.norm_pressure * 0.28
        + max(0.0, 0.48 - scene_state.safety_margin) * 0.34
        + repair_need * 0.18
        + scene_state.environmental_load * 0.12
    )
    containment_need = _clamp01(
        scene_state.environmental_load * 0.34
        + max(0.0, 0.52 - scene_state.safety_margin) * 0.32
        + repair_need * 0.18
    )
    shared_world_signal = _clamp01(
        relational_mood.shared_world_pull * 0.46
        + relational_mood.future_pull * 0.22
        + shared_attention * 0.16
        + max(0.0, scene_state.privacy_level - 0.4) * 0.1
    )
    future_signal = _clamp01(relational_mood.future_pull)
    care_signal = _clamp01(relational_mood.care)
    reverence_signal = _clamp01(relational_mood.reverence + scene_state.norm_pressure * 0.2)
    reflection_need = _clamp01(
        live_regulation.fantasy_loop_pull * 0.24
        + live_regulation.past_loop_pull * 0.32
        + max(0.0, 1.0 - relational_mood.confidence_signal) * 0.18
    )
    ambiguity_tolerance = _clamp01(
        scene_state.privacy_level * 0.18
        + max(0.0, 1.0 - scene_state.norm_pressure) * 0.18
        + relational_mood.innocence * 0.18
    )
    coordination_bonus = 0.14 if scene_state.task_phase in {"coordination", "co_work", "shared_task"} else 0.0
    clarify_need = _clamp01(
        live_regulation.future_loop_pull * 0.14
        + max(0.0, 0.45 - shared_attention) * 0.3
        + max(0.0, 0.42 - relational_mood.confidence_signal) * 0.22
        + coordination_bonus
    )
    withdraw_need = _clamp01(
        boundary_need * 0.38
        + scene_state.environmental_load * 0.24
        + max(0.0, 0.3 - scene_state.safety_margin) * 0.28
    )

    return {
        "shared_attention": shared_attention,
        "repair_need": repair_need,
        "contact_readiness": contact_readiness,
        "boundary_need": boundary_need,
        "containment_need": containment_need,
        "shared_world_signal": shared_world_signal,
        "future_signal": future_signal,
        "care_signal": care_signal,
        "reverence_signal": reverence_signal,
        "reflection_need": reflection_need,
        "ambiguity_tolerance": ambiguity_tolerance,
        "clarity_need": clarify_need,
        "withdraw_need": withdraw_need,
        "privacy_room": _clamp01(scene_state.privacy_level),
        "norm_pressure": _clamp01(scene_state.norm_pressure),
        "norm_room": _clamp01(1.0 - scene_state.norm_pressure),
        "constraint_body_cost": _clamp01((constraint_field or {}).get("body_cost", 0.0)),
        "constraint_boundary": _clamp01((constraint_field or {}).get("boundary_pressure", 0.0)),
        "constraint_repair": _clamp01((constraint_field or {}).get("repair_pressure", 0.0)),
        "constraint_shared_world": _clamp01((constraint_field or {}).get("shared_world_pressure", 0.0)),
    }


def _resolve_disclosure_depth(
    *,
    preferred: str,
    scene_state: SceneState,
    live_regulation: LiveInteractionRegulation,
) -> str:
    if scene_state.norm_pressure >= 0.66 or live_regulation.strained_pause >= 0.56:
        return "minimal"
    if preferred == "medium" and scene_state.privacy_level >= 0.54:
        return "medium"
    return preferred


def _resolve_timing_mode(
    *,
    preferred: str,
    scene_state: SceneState,
    live_regulation: LiveInteractionRegulation,
) -> str:
    if live_regulation.repair_window_open and "repair" not in preferred:
        return "repair_sensitive_return"
    if scene_state.temporal_phase in {"parting", "arrival"}:
        return f"{preferred}_phase"
    return preferred


def _resolve_boundary_mode(
    *,
    preferred: str,
    scene_state: SceneState,
    live_regulation: LiveInteractionRegulation,
) -> str:
    if scene_state.norm_pressure >= 0.72 and preferred == "permeable":
        return "respectful"
    if live_regulation.strained_pause >= 0.58 and preferred not in {"protective", "closed_boundary"}:
        return "softened"
    return preferred


def _softmax(values: Iterable[float], *, temperature: float = 1.0) -> list[float]:
    values = [float(value) for value in values]
    if not values:
        return []
    temperature = max(0.35, float(temperature))
    max_value = max(values)
    exps = [math.exp((value - max_value) / temperature) for value in values]
    denom = sum(exps) or 1.0
    return [value / denom for value in exps]


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))
