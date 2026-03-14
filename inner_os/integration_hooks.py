from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Mapping, Optional

from .conscious_access import ConsciousAccessCore, ConsciousAccessSnapshot
from .hook_contracts import (
    MemoryRecallInput,
    PostTurnUpdateInput,
    PreTurnUpdateInput,
    ResponseGateInput,
)
from .memory_core import MemoryCore
from .development_core import DevelopmentCore
from .reinterpretation_core import ReinterpretationCore
from .environment_pressure_core import EnvironmentPressureCore
from .relationship_core import RelationshipCore
from .personality_core import PersonalityIndexCore
from .persistence_core import PersistenceCore
from .physiology import BoundaryCore, PainStressCore, RecoveryCore
from .temporal import TemporalWeightCore
from .object_relation_core import ObjectRelationCore
from .peripersonal_core import PeripersonalCore
from .memory_orchestration_core import MemoryOrchestrationCore
from .forgetting_core import ForgettingCore
from .relational_world import RelationalWorldCore
from .terrain_core import AffectiveTerrainCore


EXPRESSION_HINT_WEIGHTS = {
    "tentative_roughness": 0.62,
    "tentative_caution": 0.18,
    "tentative_recent_strain": 0.12,
    "tentative_anticipation": 0.12,
    "tentative_inertia": 0.08,
    "tentative_continuity_relief": 0.08,
    "tentative_clarity_relief": 0.06,
    "assertive_roughness": 0.55,
    "assertive_caution": 0.12,
    "assertive_anticipation": 0.18,
    "question_identity_gap": 0.8,
    "question_social_gap": 0.45,
    "question_afterglow": 0.32,
    "question_anticipation": 0.24,
    "question_stabilization": 0.18,
}

CORE_EVOLUTION_WEIGHTS = {
    "decay_base": 0.22,
    "decay_social_grounding": 0.16,
    "decay_continuity": 0.12,
    "decay_stress_penalty": 0.08,
    "reinforcement_recall": 0.16,
    "reinforcement_afterglow": 0.18,
    "reinforcement_stress": 0.06,
    "reopening_continuity": 0.22,
    "reopening_social_grounding": 0.18,
    "reopening_recovery_penalty": 0.2,
    "reopening_stress_penalty": 0.12,
    "replay_current": 0.58,
    "replay_reinforcement": 0.18,
    "replay_previous": 0.34,
    "replay_decay": 0.14,
    "replay_reopening_relief": 0.08,
    "anticipation_current": 0.62,
    "anticipation_previous": 0.28,
    "anticipation_decay": 0.14,
    "anticipation_reopening_relief": 0.1,
    "stabilization_current": 0.6,
    "stabilization_previous": 0.26,
    "stabilization_decay": 0.12,
    "stabilization_recovery": 0.04,
    "clarity_current": 0.66,
    "clarity_previous": 0.22,
    "clarity_decay": 0.08,
    "clarity_reopening": 0.08,
    "inertia_current": 0.6,
    "inertia_previous": 0.3,
    "inertia_decay": 0.12,
    "inertia_reopening_relief": 0.08,
}

CORE_AXIS_WEIGHTS = {
    "replay_afterglow": 0.22,
    "replay_roughness": 0.16,
    "replay_recent_strain": 0.12,
    "replay_discontinuity": 0.1,
    "replay_recall_active": 0.18,
    "anticipation_temporal": 0.34,
    "anticipation_stress": 0.26,
    "anticipation_recovery_need": 0.14,
    "anticipation_future_signal": 0.32,
    "anticipation_roughness": 0.12,
    "anticipation_recent_strain": 0.1,
    "anticipation_continuity_relief": 0.06,
    "stabilization_recovery_need": 0.28,
    "stabilization_roughness": 0.24,
    "stabilization_anticipation": 0.26,
    "stabilization_afterglow": 0.18,
    "stabilization_transition": 0.12,
    "stabilization_social_grounding_relief": 0.08,
    "clarity_trust": 0.26,
    "clarity_continuity": 0.22,
    "clarity_social_grounding": 0.2,
    "clarity_community_resonance": 0.14,
    "clarity_culture_resonance": 0.08,
    "clarity_roughness_penalty": 0.12,
    "clarity_recent_strain_penalty": 0.08,
    "clarity_transition_penalty": 0.1,
    "inertia_roughness": 0.28,
    "inertia_afterglow": 0.18,
    "inertia_anticipation": 0.22,
    "inertia_recent_strain": 0.1,
    "inertia_clarity_gap": 0.14,
    "inertia_continuity_relief": 0.06,
}

RESPONSE_GATE_WEIGHTS = {
    "hesitation_stress": 0.45,
    "hesitation_recovery_need": 0.4,
    "hesitation_norm": 0.3,
    "hesitation_caution": 0.25,
    "hesitation_recent_strain": 0.18,
    "hesitation_roughness": 0.42,
    "hesitation_recalled_tentative": 0.16,
    "hesitation_anticipation": 0.18,
    "hesitation_stabilization": 0.14,
    "hesitation_inertia": 0.12,
    "hesitation_trust_relief": 0.08,
    "hesitation_continuity_relief": 0.06,
    "hesitation_social_grounding_relief": 0.05,
    "hesitation_community_relief": 0.05,
    "hesitation_clarity_relief": 0.08,
    "private_high_arousal_floor": 0.55,
    "autonomic_floor_threshold": 0.42,
    "autonomic_floor_value": 0.42,
    "surface_floor": 0.15,
    "surface_affiliation": 0.08,
    "surface_culture": 0.04,
    "surface_community": 0.06,
    "surface_clarity": 0.04,
    "surface_roughness_penalty": 0.06,
    "surface_recalled_tentative_penalty": 0.04,
    "surface_anticipation_penalty": 0.04,
}

RESPONSE_ALLOCATION_WEIGHTS = {
    "hold_relief_sum_cap": 0.24,
    "express_support_cap": 0.26,
    "express_penalty_cap": 0.18,
    "express_base_scale": 0.78,
}

INTENT_THRESHOLDS = {
    "identity_clarify": 0.66,
    "social_check_in": 0.72,
    "afterglow_redirect": 0.28,
    "anticipation_clarify": 0.42,
    "stabilization_check_in": 0.44,
    "tentative_definitive_cutoff": 0.28,
}

AFTERGLOW_WEIGHTS = {
    "surface_active": 0.18,
    "clarify_bonus": 0.08,
    "check_in_bonus": 0.12,
    "social_gap": 0.16,
    "identity_gap": 0.2,
    "roughness": 0.06,
}

CONTEXT_SHIFT_WEIGHTS = {
    "culture_change": 0.35,
    "community_change": 0.45,
    "role_change": 0.2,
    "place_change": 0.18,
    "body_change": 0.14,
    "privacy_shift": 0.12,
    "density_shift": 0.1,
    "social_integration": 0.55,
    "situational_integration": 0.3,
    "bodily_integration": 0.15,
}


@dataclass
class HookState:
    stress: float = 0.0
    recovery_need: float = 0.0
    attention_density: float = 0.0
    safety_bias: float = 0.0
    temporal_pressure: float = 0.0
    memory_anchor: Optional[str] = None
    replay_active: bool = False
    route: str = "reflex"
    talk_mode: str = "watch"
    body_state_flag: str = "normal"
    voice_level: float = 0.0
    person_count: int = 0
    autonomic_balance: float = 0.5
    belonging: float = 0.45
    trust_bias: float = 0.45
    norm_pressure: float = 0.35
    role_commitment: float = 0.4
    attachment: float = 0.42
    trust_memory: float = 0.45
    familiarity: float = 0.35
    role_alignment: float = 0.4
    rupture_sensitivity: float = 0.38
    caution_bias: float = 0.4
    affiliation_bias: float = 0.45
    exploration_bias: float = 0.4
    reflective_bias: float = 0.45
    continuity_score: float = 0.48
    social_grounding: float = 0.44
    recent_strain: float = 0.32
    culture_resonance: float = 0.0
    community_resonance: float = 0.0
    terrain_transition_roughness: float = 0.0
    recalled_tentative_bias: float = 0.0
    social_update_strength: float = 1.0
    identity_update_strength: float = 1.0
    interaction_afterglow: float = 0.0
    interaction_afterglow_intent: Optional[str] = None
    replay_intensity: float = 0.0
    anticipation_tension: float = 0.0
    stabilization_drive: float = 0.0
    relational_clarity: float = 0.0
    meaning_inertia: float = 0.0
    recovery_reopening: float = 0.0
    object_affordance_bias: float = 0.0
    fragility_guard: float = 0.0
    object_attachment: float = 0.0
    object_avoidance: float = 0.0
    tool_extension_bias: float = 0.0
    ritually_sensitive_bias: float = 0.0
    reachability: float = 0.0
    near_body_risk: float = 0.0
    defensive_salience: float = 0.0
    approach_confidence: float = 0.0
    reuse_trajectory: float = 0.0
    interference_pressure: float = 0.0
    consolidation_priority: float = 0.0
    prospective_memory_pull: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreTurnUpdateResult:
    state: HookState
    interaction_hints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "interaction_hints": dict(self.interaction_hints),
        }


@dataclass
class MemoryRecallResult:
    recall_payload: Dict[str, Any] = field(default_factory=dict)
    retrieval_summary: Dict[str, Any] = field(default_factory=dict)
    ignition_hints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recall_payload": dict(self.recall_payload),
            "retrieval_summary": dict(self.retrieval_summary),
            "ignition_hints": dict(self.ignition_hints),
        }


@dataclass
class ResponseGateResult:
    talk_mode: str
    route: str
    allowed_surface_intensity: float
    hesitation_bias: float
    conscious_access: Dict[str, Any] = field(default_factory=dict)
    expression_hints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "talk_mode": self.talk_mode,
            "route": self.route,
            "allowed_surface_intensity": float(self.allowed_surface_intensity),
            "hesitation_bias": float(self.hesitation_bias),
            "conscious_access": dict(self.conscious_access),
            "expression_hints": dict(self.expression_hints),
        }


@dataclass
class PostTurnUpdateResult:
    state: HookState
    memory_appends: list[Dict[str, Any]] = field(default_factory=list)
    audit_record: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "memory_appends": list(self.memory_appends),
            "audit_record": dict(self.audit_record),
        }


class IntegrationHooks:
    """Small injectable hooks for existing systems."""

    def __init__(
        self,
        *,
        pain_stress_core: Optional[PainStressCore] = None,
        recovery_core: Optional[RecoveryCore] = None,
        boundary_core: Optional[BoundaryCore] = None,
        conscious_access_core: Optional[ConsciousAccessCore] = None,
        temporal_core: Optional[TemporalWeightCore] = None,
        memory_core: Optional[MemoryCore] = None,
        terrain_core: Optional[AffectiveTerrainCore] = None,
        relational_world_core: Optional[RelationalWorldCore] = None,
        development_core: Optional[DevelopmentCore] = None,
        reinterpretation_core: Optional[ReinterpretationCore] = None,
        environment_pressure_core: Optional[EnvironmentPressureCore] = None,
        relationship_core: Optional[RelationshipCore] = None,
        personality_core: Optional[PersonalityIndexCore] = None,
        persistence_core: Optional[PersistenceCore] = None,
        forgetting_core: Optional[ForgettingCore] = None,
        forgetting_controller: Optional[Any] = None,
        object_relation_core: Optional[ObjectRelationCore] = None,
        peripersonal_core: Optional[PeripersonalCore] = None,
        memory_orchestration_core: Optional[MemoryOrchestrationCore] = None,
    ) -> None:
        self.pain_stress_core = pain_stress_core or PainStressCore()
        self.recovery_core = recovery_core or RecoveryCore()
        self.boundary_core = boundary_core or BoundaryCore()
        self.conscious_access_core = conscious_access_core or ConsciousAccessCore()
        self.temporal_core = temporal_core or TemporalWeightCore()
        self.memory_core = memory_core or MemoryCore()
        self.terrain_core = terrain_core or AffectiveTerrainCore()
        self.relational_world_core = relational_world_core or RelationalWorldCore()
        self.development_core = development_core or DevelopmentCore()
        self.reinterpretation_core = reinterpretation_core or ReinterpretationCore()
        self.environment_pressure_core = environment_pressure_core or EnvironmentPressureCore()
        self.relationship_core = relationship_core or RelationshipCore()
        self.personality_core = personality_core or PersonalityIndexCore()
        self.persistence_core = persistence_core or PersistenceCore()
        self.forgetting_core = forgetting_core or ForgettingCore(forgetting_controller)
        self.object_relation_core = object_relation_core or ObjectRelationCore()
        self.peripersonal_core = peripersonal_core or PeripersonalCore()
        self.memory_orchestration_core = memory_orchestration_core or MemoryOrchestrationCore()

    def pre_turn_update(
        self,
        *,
        user_input: Mapping[str, Any],
        sensor_input: Mapping[str, Any],
        local_context: Mapping[str, Any],
        current_state: Optional[Mapping[str, Any]] = None,
        safety_bias: float = 0.0,
    ) -> PreTurnUpdateResult:
        gate_context = _mapping_or_empty(local_context.get("last_gate_context"))
        world_context = self.relational_world_core.absorb_context(
            _mapping_or_empty(local_context.get("relational_world"))
        )
        identity_trace = self.memory_core.load_latest_identity_trace(
            culture_id=str(world_context.get("culture_id") or "") or None,
            community_id=str(world_context.get("community_id") or "") or None,
            social_role=str(world_context.get("social_role") or "") or None,
        )
        relationship_trace = self.memory_core.load_latest_profile_record(
            kind="relationship_trace",
            culture_id=str(world_context.get("culture_id") or "") or None,
            community_id=str(world_context.get("community_id") or "") or None,
            social_role=str(world_context.get("social_role") or "") or None,
            memory_anchor=str(world_context.get("place_memory_anchor") or "") or None,
        )
        community_profile_trace = self.memory_core.load_latest_profile_record(
            kind="community_profile_trace",
            culture_id=str(world_context.get("culture_id") or "") or None,
            community_id=str(world_context.get("community_id") or "") or None,
        )
        merged_state = dict(identity_trace or {})
        merged_state.update(dict(relationship_trace or {}))
        merged_state.update(dict(community_profile_trace or {}))
        merged_state.update(dict(current_state or {}))
        transition_signal = _context_shift_signal(
            current_state=current_state,
            relational_world=world_context,
            identity_trace=identity_trace,
            community_profile_trace=community_profile_trace,
            sensor_input=sensor_input,
        )
        if transition_signal["transition_intensity"] > 0.0:
            merged_state["recent_strain"] = _float_from(merged_state, "recent_strain", default=0.32) + transition_signal["transition_intensity"] * 0.12
            merged_state["community_resonance"] = max(0.0, _float_from(merged_state, "community_resonance", default=0.0) - transition_signal["transition_intensity"] * 0.12)
            merged_state["culture_resonance"] = max(0.0, _float_from(merged_state, "culture_resonance", default=0.0) - transition_signal["transition_intensity"] * 0.08)
        previous_energy = _float_from(merged_state, "current_energy", default=0.7)
        previous_pressure = _float_from(merged_state, "temporal_pressure", default=0.0)
        stress = self.pain_stress_core.stress(
            sensor_metrics=sensor_input,
            last_shadow_estimate=_mapping_or_none(local_context.get("last_shadow_estimate")),
            last_gate_context=gate_context,
        )
        environment_pressure = self.environment_pressure_core.snapshot(
            relational_world=world_context,
            sensor_input=sensor_input,
            current_state=merged_state,
        )
        relationship = self.relationship_core.snapshot(
            relational_world=world_context,
            sensor_input=sensor_input,
            current_state=merged_state,
        )
        development = self.development_core.snapshot(
            relational_world=world_context,
            sensor_input=sensor_input,
            current_state=merged_state,
            safety_bias=safety_bias,
            environment_pressure=environment_pressure.to_dict(),
        )
        development_biases = self.development_core.memory_kind_biases(
            state=development.to_dict(),
        )
        object_relation = self.object_relation_core.snapshot(
            relational_world=world_context,
            current_state=merged_state,
        )
        peripersonal = self.peripersonal_core.snapshot(
            relational_world=world_context,
            sensor_input=sensor_input,
            current_state=merged_state,
            object_relation=object_relation.to_dict(),
        )
        personality = self.personality_core.snapshot(
            current_state=merged_state,
            development=development.to_dict(),
            relationship=relationship.to_dict(),
            environment_pressure=environment_pressure.to_dict(),
        )
        persistence = self.persistence_core.snapshot(
            current_state=merged_state,
            development=development.to_dict(),
            relationship=relationship.to_dict(),
            personality=personality.to_dict(),
            environment_pressure=environment_pressure.to_dict(),
            transition_signal=transition_signal,
        )
        recovery_need = self.recovery_core.recovery_need(
            stress=stress,
            current_energy=previous_energy,
        )
        attention_density = self.recovery_core.attention_density(
            sensor_metrics=sensor_input,
            last_gate_context=gate_context,
        )
        temporal_pressure = self.temporal_core.push(
            max(stress, attention_density * 0.35, previous_pressure * 0.1)
        )
        voice_level = float(sensor_input.get("voice_level", 0.0) or 0.0)
        person_count = int(sensor_input.get("person_count", 0) or 0)
        autonomic_balance = float(sensor_input.get("autonomic_balance", 0.5) or 0.5)
        body_state_flag = str(sensor_input.get("body_state_flag") or "normal")
        privacy_tags = [str(tag).lower() for tag in (sensor_input.get("privacy_tags") or [])]
        route = "conscious" if temporal_pressure < 0.72 else "reflex"
        talk_mode = "watch" if temporal_pressure < 0.45 else "ask"
        if body_state_flag == "private_high_arousal" or "private" in privacy_tags:
            route = "conscious"
            talk_mode = "soothe"
        elif voice_level >= 0.45 and person_count > 0 and temporal_pressure < 0.72:
            talk_mode = "talk"
        if development.norm_pressure >= 0.62 and talk_mode == "talk":
            talk_mode = "ask"
        if development.belonging >= 0.58 and temporal_pressure < 0.6:
            route = "conscious"
        if personality.caution_bias >= 0.62:
            talk_mode = "soothe" if talk_mode == "talk" else talk_mode
        if personality.affiliation_bias >= 0.64 and talk_mode == "watch" and temporal_pressure < 0.62:
            talk_mode = "ask"
        terrain = self.terrain_core.snapshot(
            valence=float(gate_context.get("valence", 0.0) or 0.0),
            arousal=float(gate_context.get("arousal", 0.0) or 0.0),
            stress=stress,
            temporal_pressure=temporal_pressure,
            transition_intensity=_float_from(transition_signal, "transition_intensity", default=0.0),
            social_grounding=persistence.social_grounding,
            community_resonance=persistence.community_resonance,
        )
        if terrain.transition_roughness >= 0.38 and talk_mode == "talk":
            talk_mode = "ask"
        future_signal = _float_from(current_state, "future_signal", default=0.0)
        raw_core_axes = _core_state_axes(
            stress=stress,
            recovery_need=recovery_need,
            temporal_pressure=temporal_pressure,
            continuity_score=persistence.continuity_score,
            social_grounding=persistence.social_grounding,
            recent_strain=persistence.recent_strain,
            trust_bias=development.trust_bias,
            terrain_transition_roughness=terrain.transition_roughness,
            interaction_afterglow=_float_from(current_state, "interaction_afterglow", default=0.0),
            transition_intensity=_float_from(transition_signal, "transition_intensity", default=0.0),
            community_resonance=persistence.community_resonance,
            culture_resonance=persistence.culture_resonance,
            future_signal=future_signal,
            recall_active=False,
        )
        core_axes = _evolve_core_axes(
            previous=current_state,
            current=raw_core_axes,
            stress=stress,
            recovery_need=recovery_need,
            continuity_score=persistence.continuity_score,
            social_grounding=persistence.social_grounding,
            recall_active=False,
            interaction_afterglow=_float_from(current_state, "interaction_afterglow", default=0.0),
        )
        prior_identity_update_strength = _float_from(current_state, "identity_update_strength", default=1.0)
        prior_social_update_strength = _float_from(current_state, "social_update_strength", default=1.0)
        prior_interaction_afterglow = _float_from(current_state, "interaction_afterglow", default=0.0)
        prior_afterglow_intent = _text_or_none((current_state or {}).get("interaction_afterglow_intent"))
        if prior_identity_update_strength <= 0.68 and talk_mode in {"talk", "watch"} and temporal_pressure < 0.72:
            talk_mode = "ask"
            route = "conscious"
        if prior_social_update_strength <= 0.72 and talk_mode == "watch" and temporal_pressure < 0.58:
            talk_mode = "ask"
        if prior_interaction_afterglow >= 0.24 and talk_mode in {"watch", "talk"} and temporal_pressure < 0.64:
            talk_mode = "ask"
            route = "conscious"
        if core_axes["stabilization_drive"] >= 0.42 and talk_mode == "talk":
            talk_mode = "ask"
        if core_axes["anticipation_tension"] >= 0.4:
            route = "conscious"
        if terrain.attractor == "unfamiliar_slope" and talk_mode != "soothe":
            route = "conscious"
            talk_mode = "watch"
        expression_hints = _expression_hints(
            terrain_transition_roughness=terrain.transition_roughness,
            caution_bias=personality.caution_bias,
            recent_strain=persistence.recent_strain,
            continuity_score=persistence.continuity_score,
            social_update_strength=_float_from(current_state, "social_update_strength", default=1.0),
            identity_update_strength=_float_from(current_state, "identity_update_strength", default=1.0),
            interaction_afterglow=prior_interaction_afterglow,
            interaction_afterglow_intent=prior_afterglow_intent,
            replay_intensity=core_axes["replay_intensity"],
            anticipation_tension=core_axes["anticipation_tension"],
            stabilization_drive=core_axes["stabilization_drive"],
            relational_clarity=core_axes["relational_clarity"],
            meaning_inertia=core_axes["meaning_inertia"],
            recovery_reopening=core_axes["recovery_reopening"],
            object_affordance_bias=object_relation.object_affordance_bias,
            fragility_guard=object_relation.fragility_guard,
            object_attachment=object_relation.object_attachment,
            object_avoidance=object_relation.object_avoidance,
            tool_extension_bias=object_relation.tool_extension_bias,
            ritually_sensitive_bias=object_relation.ritually_sensitive_bias,
            defensive_salience=peripersonal.defensive_salience,
            reachability=peripersonal.reachability,
        )
        state = HookState(
            stress=stress,
            recovery_need=recovery_need,
            attention_density=attention_density,
            safety_bias=float(max(safety_bias, 0.0)),
            temporal_pressure=temporal_pressure,
            route=route,
            talk_mode=talk_mode,
            body_state_flag=body_state_flag,
            voice_level=voice_level,
            person_count=person_count,
            autonomic_balance=autonomic_balance,
            belonging=development.belonging,
            trust_bias=development.trust_bias,
            norm_pressure=development.norm_pressure,
            role_commitment=development.role_commitment,
            attachment=relationship.attachment,
            trust_memory=relationship.trust_memory,
            familiarity=relationship.familiarity,
            role_alignment=relationship.role_alignment,
            rupture_sensitivity=relationship.rupture_sensitivity,
            caution_bias=personality.caution_bias,
            affiliation_bias=personality.affiliation_bias,
            exploration_bias=personality.exploration_bias,
            reflective_bias=personality.reflective_bias,
            continuity_score=persistence.continuity_score,
            social_grounding=persistence.social_grounding,
            recent_strain=persistence.recent_strain,
            culture_resonance=persistence.culture_resonance,
            community_resonance=persistence.community_resonance,
            terrain_transition_roughness=terrain.transition_roughness,
            social_update_strength=prior_social_update_strength,
            identity_update_strength=prior_identity_update_strength,
            interaction_afterglow=prior_interaction_afterglow,
            interaction_afterglow_intent=prior_afterglow_intent,
            replay_intensity=core_axes["replay_intensity"],
            anticipation_tension=core_axes["anticipation_tension"],
            stabilization_drive=core_axes["stabilization_drive"],
            relational_clarity=core_axes["relational_clarity"],
            meaning_inertia=core_axes["meaning_inertia"],
            recovery_reopening=core_axes["recovery_reopening"],
            object_affordance_bias=object_relation.object_affordance_bias,
            fragility_guard=object_relation.fragility_guard,
            object_attachment=object_relation.object_attachment,
            object_avoidance=object_relation.object_avoidance,
            tool_extension_bias=object_relation.tool_extension_bias,
            ritually_sensitive_bias=object_relation.ritually_sensitive_bias,
            defensive_salience=peripersonal.defensive_salience,
            reachability=peripersonal.reachability,
        )
        return PreTurnUpdateResult(
            state=state,
            interaction_hints={
                "surface_route": route,
                "suggested_talk_mode": talk_mode,
                "watchful": temporal_pressure >= 0.45,
                "input_text": str(user_input.get("text") or "").strip(),
                "terrain": terrain.to_dict(),
                "voice_level": round(voice_level, 4),
                "person_count": person_count,
                "autonomic_balance": round(autonomic_balance, 4),
                "body_state_flag": body_state_flag,
                "relational_world": world_context,
                "development": development.to_dict(),
                "development_biases": development_biases,
                "relationship": relationship.to_dict(),
                "personality": personality.to_dict(),
                "persistence": persistence.to_dict(),
                "relationship_trace": relationship_trace,
                "community_profile_trace": community_profile_trace,
                "expression_hint": expression_hints,
                "environment_pressure": environment_pressure.to_dict(),
                "transition_signal": transition_signal,
                "context_shift": transition_signal,
                "identity_trace": identity_trace,
                "interaction_afterglow": round(prior_interaction_afterglow, 4),
                "interaction_afterglow_intent": prior_afterglow_intent,
                "core_state": dict(core_axes),
                "raw_core_state": dict(raw_core_axes),
                "object_relation": object_relation.to_dict(),
                "peripersonal": peripersonal.to_dict(),
            },
        )

    def memory_recall(
        self,
        *,
        text_cue: str = "",
        visual_cue: str = "",
        world_cue: str = "",
        current_state: Optional[Mapping[str, Any]] = None,
        retrieval_summary: Optional[Mapping[str, Any]] = None,
    ) -> MemoryRecallResult:
        world_snapshot = self.relational_world_core.snapshot()
        world_anchor = str(world_snapshot.get("place_memory_anchor") or "").strip()
        nearby_objects = " ".join(world_snapshot.get("nearby_objects") or [])
        cue_text = " ".join(
            part.strip()
            for part in (text_cue, visual_cue, world_cue, world_anchor, nearby_objects)
            if str(part).strip()
        ).strip()
        if not cue_text:
            return MemoryRecallResult()

        cue_strength = min(1.0, max(len(cue_text) / 120.0, 0.08))
        ignition = self.temporal_core.reignite(cue_strength)
        retrieval = dict(retrieval_summary or {})

        identity_trace = self.memory_core.load_latest_identity_trace(
            culture_id=str(world_snapshot.get("culture_id") or "") or None,
            community_id=str(world_snapshot.get("community_id") or "") or None,
            social_role=str(world_snapshot.get("social_role") or "") or None,
        )
        relationship_trace = self.memory_core.load_latest_profile_record(
            kind="relationship_trace",
            culture_id=str(world_snapshot.get("culture_id") or "") or None,
            community_id=str(world_snapshot.get("community_id") or "") or None,
            social_role=str(world_snapshot.get("social_role") or "") or None,
            memory_anchor=str(world_snapshot.get("place_memory_anchor") or "") or None,
        )
        community_profile_trace = self.memory_core.load_latest_profile_record(
            kind="community_profile_trace",
            culture_id=str(world_snapshot.get("culture_id") or "") or None,
            community_id=str(world_snapshot.get("community_id") or "") or None,
        )
        merged_state = dict(identity_trace or {})
        merged_state.update(dict(relationship_trace or {}))
        merged_state.update(dict(community_profile_trace or {}))
        merged_state.update(dict(current_state or {}))
        transition_signal = _context_shift_signal(
            current_state=current_state,
            relational_world=world_snapshot,
            identity_trace=identity_trace,
            community_profile_trace=community_profile_trace,
            sensor_input=current_state,
        )
        if transition_signal["transition_intensity"] > 0.0:
            merged_state["recent_strain"] = _float_from(merged_state, "recent_strain", default=0.32) + transition_signal["transition_intensity"] * 0.12
            merged_state["community_resonance"] = max(0.0, _float_from(merged_state, "community_resonance", default=0.0) - transition_signal["transition_intensity"] * 0.12)
            merged_state["culture_resonance"] = max(0.0, _float_from(merged_state, "culture_resonance", default=0.0) - transition_signal["transition_intensity"] * 0.08)

        environment_pressure = self.environment_pressure_core.snapshot(
            relational_world=world_snapshot,
            sensor_input=current_state,
            current_state=merged_state,
        )
        relationship = self.relationship_core.snapshot(
            relational_world=world_snapshot,
            sensor_input=current_state,
            current_state=merged_state,
        )
        development = self.development_core.snapshot(
            relational_world=world_snapshot,
            sensor_input=current_state,
            current_state=merged_state,
            safety_bias=_float_from(merged_state, "safety_bias", default=0.0),
            environment_pressure=environment_pressure.to_dict(),
        )
        personality = self.personality_core.snapshot(
            current_state=merged_state,
            development=development.to_dict(),
            relationship=relationship.to_dict(),
            environment_pressure=environment_pressure.to_dict(),
        )
        persistence = self.persistence_core.snapshot(
            current_state=merged_state,
            development=development.to_dict(),
            relationship=relationship.to_dict(),
            personality=personality.to_dict(),
            environment_pressure=environment_pressure.to_dict(),
            transition_signal=transition_signal,
        )
        terrain = self.terrain_core.snapshot(
            valence=_float_from(merged_state, "valence", default=0.0),
            arousal=_float_from(merged_state, "arousal", default=0.0),
            stress=_float_from(merged_state, "stress", default=0.0),
            temporal_pressure=_float_from(merged_state, "temporal_pressure", default=0.0),
            memory_ignition=float(ignition),
            transition_intensity=_float_from(transition_signal, "transition_intensity", default=0.0),
            social_grounding=persistence.social_grounding,
            community_resonance=persistence.community_resonance,
        )

        development_biases = self.development_core.memory_kind_biases(
            state=development.to_dict(),
        )
        object_relation = self.object_relation_core.snapshot(
            relational_world=world_snapshot,
            current_state=current_state,
        )
        future_signal = _float_from(current_state, "future_signal", default=0.0)
        raw_core_axes = _core_state_axes(
            stress=_float_from(merged_state, "stress", default=0.0),
            recovery_need=_float_from(merged_state, "recovery_need", default=0.0),
            temporal_pressure=_float_from(merged_state, "temporal_pressure", default=0.0),
            continuity_score=persistence.continuity_score,
            social_grounding=persistence.social_grounding,
            recent_strain=persistence.recent_strain,
            trust_bias=development.trust_bias,
            terrain_transition_roughness=terrain.transition_roughness,
            interaction_afterglow=_float_from(current_state, "interaction_afterglow", default=0.0),
            transition_intensity=_float_from(transition_signal, "transition_intensity", default=0.0),
            community_resonance=persistence.community_resonance,
            culture_resonance=persistence.culture_resonance,
            future_signal=future_signal,
            recall_active=True,
        )
        core_axes = _evolve_core_axes(
            previous=current_state,
            current=raw_core_axes,
            stress=_float_from(merged_state, "stress", default=0.0),
            recovery_need=_float_from(merged_state, "recovery_need", default=0.0),
            continuity_score=persistence.continuity_score,
            social_grounding=persistence.social_grounding,
            recall_active=True,
            interaction_afterglow=_float_from(current_state, "interaction_afterglow", default=0.0),
        )
        merged_state.update(core_axes)
        merged_state["terrain_transition_roughness"] = terrain.transition_roughness
        forgetting_snapshot = self.forgetting_core.snapshot(
            stress=_float_from(merged_state, "stress", default=0.0),
            recovery_need=_float_from(merged_state, "recovery_need", default=0.0),
            terrain_transition_roughness=terrain.transition_roughness,
            transition_intensity=_float_from(transition_signal, "transition_intensity", default=0.0),
            recent_strain=persistence.recent_strain,
        ).to_dict()
        memory_orchestration = self.memory_orchestration_core.snapshot(
            relational_world=world_snapshot,
            current_state=current_state,
            forgetting_snapshot=forgetting_snapshot,
            recall_active=True,
        ).to_dict()
        object_relation = self.object_relation_core.snapshot(
            relational_world=world_snapshot,
            current_state=merged_state,
        )
        peripersonal = self.peripersonal_core.snapshot(
            relational_world=world_snapshot,
            current_state=merged_state,
            object_relation=object_relation.to_dict(),
        )
        local_payload = self.memory_core.build_recall_payload(
            cue_text,
            limit=3,
            bias_context={
                "culture_id": world_snapshot.get("culture_id"),
                "community_id": world_snapshot.get("community_id"),
                "social_role": world_snapshot.get("social_role"),
                "memory_anchor": world_snapshot.get("place_memory_anchor")
                or (current_state.get("memory_anchor") if isinstance(current_state, Mapping) else None),
                "caution_bias": personality.caution_bias,
                "affiliation_bias": personality.affiliation_bias,
                "continuity_score": persistence.continuity_score,
                "hazard_pressure": environment_pressure.hazard_pressure,
                "institutional_pressure": environment_pressure.institutional_pressure,
                "ritual_pressure": environment_pressure.ritual_pressure,
                "resource_pressure": environment_pressure.resource_pressure,
                "culture_resonance": persistence.culture_resonance,
                "community_resonance": persistence.community_resonance,
                "terrain_transition_roughness": terrain.transition_roughness,
                "transition_intensity": transition_signal["transition_intensity"],
                "replay_intensity": core_axes["replay_intensity"],
                "anticipation_tension": core_axes["anticipation_tension"],
                "stabilization_drive": core_axes["stabilization_drive"],
                "relational_clarity": core_axes["relational_clarity"],
                "meaning_inertia": core_axes["meaning_inertia"],
                "recovery_reopening": core_axes["recovery_reopening"],
                "forgetting_pressure": forgetting_snapshot["forgetting_pressure"],
                "replay_horizon": forgetting_snapshot["replay_horizon"],
                "monument_salience": memory_orchestration["monument_salience"],
                "monument_kind": memory_orchestration["monument_kind"],
                "conscious_mosaic_density": memory_orchestration["conscious_mosaic_density"],
                "object_affordance_bias": object_relation.object_affordance_bias,
                "fragility_guard": object_relation.fragility_guard,
                "object_attachment": object_relation.object_attachment,
                "object_avoidance": object_relation.object_avoidance,
                "tool_extension_bias": object_relation.tool_extension_bias,
                "ritually_sensitive_bias": object_relation.ritually_sensitive_bias,
                "reachability": peripersonal.reachability,
                "near_body_risk": peripersonal.near_body_risk,
                "defensive_salience": peripersonal.defensive_salience,
                "approach_confidence": peripersonal.approach_confidence,
                "reuse_trajectory": memory_orchestration["reuse_trajectory"],
                "interference_pressure": memory_orchestration["interference_pressure"],
                "consolidation_priority": memory_orchestration["consolidation_priority"],
                "prospective_memory_pull": memory_orchestration["prospective_memory_pull"],
                "kind_biases": development_biases,
            },
        )
        if local_payload.get("record_id"):
            touched_record = self.memory_core.touch_record_usage(str(local_payload.get("record_id") or ""))
            if touched_record:
                local_payload["access_count"] = touched_record.get("access_count")
                local_payload["primed_weight"] = touched_record.get("primed_weight")
                local_payload["last_accessed_at"] = touched_record.get("last_accessed_at")
        if local_payload.get("hits"):
            retrieval.setdefault("inner_os_memory", local_payload.get("hits"))

        anchor = str(local_payload.get("memory_anchor") or cue_text[:160]).strip()
        recall_payload = {
            "memory_anchor": anchor,
            "cue_text": cue_text,
            "culture_id": world_snapshot.get("culture_id"),
            "community_id": world_snapshot.get("community_id"),
            "social_role": world_snapshot.get("social_role"),
        }
        if local_payload:
            recall_payload.update(
                {
                    "summary": local_payload.get("summary"),
                    "text": local_payload.get("text"),
                    "record_kind": local_payload.get("record_kind"),
                    "record_provenance": local_payload.get("record_provenance"),
                    "source_episode_id": local_payload.get("source_episode_id"),
                    "policy_hint": local_payload.get("policy_hint"),
                    "kind_breakdown": local_payload.get("kind_breakdown"),
                    "tentative_bias": local_payload.get("tentative_bias"),
                    "access_count": local_payload.get("access_count"),
                    "primed_weight": local_payload.get("primed_weight"),
                    "last_accessed_at": local_payload.get("last_accessed_at"),
                    "replay_intensity": round(core_axes["replay_intensity"], 4),
                    "anticipation_tension": round(core_axes["anticipation_tension"], 4),
                    "relational_clarity": round(core_axes["relational_clarity"], 4),
                    "meaning_inertia": round(core_axes["meaning_inertia"], 4),
                    "recovery_reopening": round(core_axes["recovery_reopening"], 4),
                    "forgetting_pressure": round(forgetting_snapshot["forgetting_pressure"], 4),
                    "replay_horizon": forgetting_snapshot["replay_horizon"],
                    "monument_salience": round(memory_orchestration["monument_salience"], 4),
                    "monument_kind": memory_orchestration["monument_kind"],
                    "conscious_mosaic_density": round(memory_orchestration["conscious_mosaic_density"], 4),
                    "object_affordance_bias": object_relation.object_affordance_bias,
                    "fragility_guard": object_relation.fragility_guard,
                    "object_attachment": object_relation.object_attachment,
                    "object_avoidance": object_relation.object_avoidance,
                    "tool_extension_bias": object_relation.tool_extension_bias,
                    "ritually_sensitive_bias": object_relation.ritually_sensitive_bias,
                    "reachability": peripersonal.reachability,
                    "near_body_risk": peripersonal.near_body_risk,
                    "defensive_salience": peripersonal.defensive_salience,
                    "approach_confidence": peripersonal.approach_confidence,
                    "reuse_trajectory": round(memory_orchestration["reuse_trajectory"], 4),
                    "interference_pressure": round(memory_orchestration["interference_pressure"], 4),
                    "consolidation_priority": round(memory_orchestration["consolidation_priority"], 4),
                    "prospective_memory_pull": round(memory_orchestration["prospective_memory_pull"], 4),
                }
            )

        reinterpretation = self.reinterpretation_core.snapshot(
            recall_payload=recall_payload,
            current_state=merged_state,
            relational_world=world_snapshot,
            environment_pressure=environment_pressure.to_dict(),
            transition_signal=transition_signal,
        )
        recall_payload.update(
            {
                "reinterpretation_mode": reinterpretation.mode,
                "reflective_tension": round(reinterpretation.reflective_tension, 4),
                "social_self_pressure": round(reinterpretation.social_self_pressure, 4),
                "meaning_shift": round(reinterpretation.meaning_shift, 4),
                "terrain_transition_roughness": round(reinterpretation.terrain_transition_roughness, 4),
                "reinterpretation_summary": reinterpretation.summary,
                "environment_summary": environment_pressure.summary,
                "caution_bias": round(personality.caution_bias, 4),
                "affiliation_bias": round(personality.affiliation_bias, 4),
                "reflective_bias": round(personality.reflective_bias, 4),
                "continuity_score": round(persistence.continuity_score, 4),
                "social_grounding": round(persistence.social_grounding, 4),
                "recent_strain": round(persistence.recent_strain, 4),
            }
        )

        return MemoryRecallResult(
            recall_payload=recall_payload,
            retrieval_summary=retrieval,
            ignition_hints={
                "cue_strength": round(cue_strength, 4),
                "temporal_ignition": round(ignition, 4),
                "recall_active": True,
                "terrain": terrain.to_dict(),
                "reinterpretation": reinterpretation.to_dict(),
                "development": development.to_dict(),
                "relationship": relationship.to_dict(),
                "personality": personality.to_dict(),
                "persistence": persistence.to_dict(),
                "forgetting": dict(forgetting_snapshot),
                "memory_orchestration": dict(memory_orchestration),
                "object_relation": object_relation.to_dict(),
                "peripersonal": peripersonal.to_dict(),
                "identity_trace": identity_trace,
                "relationship_trace": relationship_trace,
                "community_profile_trace": community_profile_trace,
                "environment_pressure": environment_pressure.to_dict(),
                "transition_signal": transition_signal,
                "context_shift": transition_signal,
                "community_profile_trace": community_profile_trace,
            },
        )

    def response_gate(
        self,
        *,
        draft: Mapping[str, Any],
        current_state: Mapping[str, Any],
        safety_signals: Mapping[str, Any],
    ) -> ResponseGateResult:
        talk_mode = str(current_state.get("talk_mode") or "watch")
        route = str(current_state.get("route") or "reflex")
        memory_anchor = _text_or_none(current_state.get("memory_anchor"))
        replay_active = bool(current_state.get("replay_active", False))
        access: ConsciousAccessSnapshot = self.conscious_access_core.snapshot(
            talk_mode=talk_mode,
            route=route,
            mode=str(current_state.get("mode") or "reality"),
            memory_anchor=memory_anchor,
            replay_active=replay_active,
            body_state_flag=str(current_state.get("body_state_flag") or "normal"),
            voice_level=_float_from(current_state, "voice_level", default=0.0),
            person_count=int(current_state.get("person_count", 0) or 0),
            autonomic_balance=_float_from(current_state, "autonomic_balance", default=0.5),
        )
        stress = _float_from(current_state, "stress", default=0.0)
        recovery_need = _float_from(current_state, "recovery_need", default=0.0)
        safety_bias = max(
            _float_from(current_state, "safety_bias", default=0.0),
            _float_from(safety_signals, "safety_bias", default=0.0),
        )
        body_state_flag = str(current_state.get("body_state_flag") or "normal")
        autonomic_balance = _float_from(current_state, "autonomic_balance", default=0.5)
        norm_pressure = _float_from(current_state, "norm_pressure", default=0.35)
        trust_bias = _float_from(current_state, "trust_bias", default=0.45)
        caution_bias = _float_from(current_state, "caution_bias", default=0.4)
        affiliation_bias = _float_from(current_state, "affiliation_bias", default=0.45)
        continuity_score = _float_from(current_state, "continuity_score", default=0.48)
        social_grounding = _float_from(current_state, "social_grounding", default=0.44)
        recent_strain = _float_from(current_state, "recent_strain", default=0.32)
        culture_resonance = _float_from(current_state, "culture_resonance", default=0.0)
        community_resonance = _float_from(current_state, "community_resonance", default=0.0)
        terrain_transition_roughness = _float_from(current_state, "terrain_transition_roughness", default=0.0)
        recalled_tentative_bias = _float_from(current_state, "recalled_tentative_bias", default=0.0)
        social_update_strength = _float_from(current_state, "social_update_strength", default=1.0)
        identity_update_strength = _float_from(current_state, "identity_update_strength", default=1.0)
        interaction_afterglow = _float_from(current_state, "interaction_afterglow", default=0.0)
        interaction_afterglow_intent = _text_or_none((current_state or {}).get("interaction_afterglow_intent"))
        replay_intensity = _float_from(current_state, "replay_intensity", default=0.0)
        anticipation_tension = _float_from(current_state, "anticipation_tension", default=0.0)
        stabilization_drive = _float_from(current_state, "stabilization_drive", default=0.0)
        relational_clarity = _float_from(current_state, "relational_clarity", default=0.0)
        meaning_inertia = _float_from(current_state, "meaning_inertia", default=0.0)
        object_affordance_bias = _float_from(current_state, "object_affordance_bias", default=0.0)
        fragility_guard = _float_from(current_state, "fragility_guard", default=0.0)
        object_attachment = _float_from(current_state, "object_attachment", default=0.0)
        object_avoidance = _float_from(current_state, "object_avoidance", default=0.0)
        ritually_sensitive_bias = _float_from(current_state, "ritually_sensitive_bias", default=0.0)
        reachability = _float_from(current_state, "reachability", default=0.0)
        near_body_risk = _float_from(current_state, "near_body_risk", default=0.0)
        defensive_salience = _float_from(current_state, "defensive_salience", default=0.0)
        approach_confidence = _float_from(current_state, "approach_confidence", default=0.0)
        response_allocation = _response_expression_allocation(
            stress=stress,
            recovery_need=recovery_need,
            safety_bias=safety_bias,
            body_state_flag=body_state_flag,
            autonomic_balance=autonomic_balance,
            norm_pressure=norm_pressure,
            trust_bias=trust_bias,
            caution_bias=caution_bias,
            affiliation_bias=affiliation_bias,
            continuity_score=continuity_score,
            social_grounding=social_grounding,
            recent_strain=recent_strain,
            culture_resonance=culture_resonance,
            community_resonance=community_resonance,
            terrain_transition_roughness=terrain_transition_roughness,
            recalled_tentative_bias=recalled_tentative_bias,
            anticipation_tension=anticipation_tension,
            stabilization_drive=stabilization_drive,
            relational_clarity=relational_clarity,
            meaning_inertia=meaning_inertia,
        )
        object_hold = _clamp01(fragility_guard * 0.22 + object_avoidance * 0.24 + ritually_sensitive_bias * 0.16 + defensive_salience * 0.2 + near_body_risk * 0.12)
        object_release = _clamp01(object_affordance_bias * 0.08 + object_attachment * 0.06 + reachability * 0.08 + approach_confidence * 0.06)
        hesitation_bias = _clamp01(float(response_allocation["hold_back"]) + object_hold - object_release)
        allowed_surface_intensity = max(RESPONSE_GATE_WEIGHTS["surface_floor"], float(response_allocation["express_now"]) - object_hold * 0.12 + object_release * 0.08)
        expression_hints = _expression_hints(
            terrain_transition_roughness=terrain_transition_roughness,
            caution_bias=caution_bias,
            recent_strain=recent_strain,
            continuity_score=continuity_score,
            social_update_strength=social_update_strength,
            identity_update_strength=identity_update_strength,
            interaction_afterglow=interaction_afterglow,
            interaction_afterglow_intent=interaction_afterglow_intent,
            replay_intensity=replay_intensity,
            anticipation_tension=anticipation_tension,
            stabilization_drive=stabilization_drive,
            relational_clarity=relational_clarity,
            meaning_inertia=meaning_inertia,
            recovery_reopening=_float_from(current_state, "recovery_reopening", default=0.0),
            object_affordance_bias=object_affordance_bias,
            fragility_guard=fragility_guard,
            object_attachment=object_attachment,
            object_avoidance=object_avoidance,
            tool_extension_bias=_float_from(current_state, "tool_extension_bias", default=0.0),
            ritually_sensitive_bias=ritually_sensitive_bias,
            defensive_salience=defensive_salience,
            reachability=reachability,
        )
        expression_hints["recalled_tentative_bias"] = round(recalled_tentative_bias, 4)
        expression_hints["tentative_bias"] = round(_clamp01(max(float(expression_hints.get("tentative_bias", 0.0) or 0.0), recalled_tentative_bias * 0.9)), 4)
        expression_hints["express_now"] = round(float(response_allocation["express_now"]), 4)
        expression_hints["hold_back"] = round(float(response_allocation["hold_back"]), 4)
        if recalled_tentative_bias >= INTENT_THRESHOLDS["tentative_definitive_cutoff"]:
            expression_hints["avoid_definitive_interpretation"] = True
        access_payload = access.to_dict()
        if identity_update_strength <= INTENT_THRESHOLDS["identity_clarify"] and access_payload.get("intent") in {"engage", "remember", "answer"}:
            access_payload["intent"] = "clarify"
        elif social_update_strength <= INTENT_THRESHOLDS["social_check_in"] and access_payload.get("intent") == "listen":
            access_payload["intent"] = "check_in"
        elif interaction_afterglow >= INTENT_THRESHOLDS["afterglow_redirect"] and access_payload.get("intent") in {"engage", "listen", "answer"}:
            access_payload["intent"] = "check_in" if interaction_afterglow_intent == "check_in" else "clarify"
        elif anticipation_tension >= INTENT_THRESHOLDS["anticipation_clarify"] and access_payload.get("intent") in {"engage", "answer"}:
            access_payload["intent"] = "clarify"
        elif stabilization_drive >= INTENT_THRESHOLDS["stabilization_check_in"] and access_payload.get("intent") == "listen":
            access_payload["intent"] = "check_in"
        expression_hints["social_update_strength"] = round(social_update_strength, 4)
        expression_hints["identity_update_strength"] = round(identity_update_strength, 4)
        expression_hints["interaction_afterglow"] = round(interaction_afterglow, 4)
        expression_hints["interaction_afterglow_intent"] = interaction_afterglow_intent
        expression_hints["replay_intensity"] = round(replay_intensity, 4)
        expression_hints["anticipation_tension"] = round(anticipation_tension, 4)
        expression_hints["stabilization_drive"] = round(stabilization_drive, 4)
        expression_hints["relational_clarity"] = round(relational_clarity, 4)
        expression_hints["meaning_inertia"] = round(meaning_inertia, 4)
        return ResponseGateResult(
            talk_mode=talk_mode,
            route=route,
            allowed_surface_intensity=allowed_surface_intensity,
            hesitation_bias=hesitation_bias,
            conscious_access=access_payload,
            expression_hints=expression_hints,
        )

    def post_turn_update(
        self,
        *,
        user_input: Mapping[str, Any],
        output: Mapping[str, Any],
        current_state: Mapping[str, Any],
        memory_write_candidates: Optional[list[Mapping[str, Any]]] = None,
        recall_payload: Optional[Mapping[str, Any]] = None,
        transferred_lessons: Optional[list[Mapping[str, Any]]] = None,
    ) -> PostTurnUpdateResult:
        stress = _float_from(current_state, "stress", default=0.0)
        recovery_need = _float_from(current_state, "recovery_need", default=0.0)
        reply_text = str(output.get("text") or output.get("reply_text") or "").strip()
        lingering = min(1.0, max(stress * 0.6 + recovery_need * 0.25, 0.0))
        temporal_pressure = self.temporal_core.decay_only()
        world_snapshot = self.relational_world_core.snapshot()
        identity_trace = self.memory_core.load_latest_identity_trace(
            culture_id=str(world_snapshot.get("culture_id") or "") or None,
            community_id=str(world_snapshot.get("community_id") or "") or None,
            social_role=str(world_snapshot.get("social_role") or "") or None,
        )
        relationship_trace = self.memory_core.load_latest_profile_record(
            kind="relationship_trace",
            culture_id=str(world_snapshot.get("culture_id") or "") or None,
            community_id=str(world_snapshot.get("community_id") or "") or None,
            social_role=str(world_snapshot.get("social_role") or "") or None,
            memory_anchor=str(world_snapshot.get("place_memory_anchor") or "") or None,
        )
        community_profile_trace = self.memory_core.load_latest_profile_record(
            kind="community_profile_trace",
            culture_id=str(world_snapshot.get("culture_id") or "") or None,
            community_id=str(world_snapshot.get("community_id") or "") or None,
        )
        merged_state = dict(identity_trace or {})
        merged_state.update(dict(relationship_trace or {}))
        merged_state.update(dict(community_profile_trace or {}))
        merged_state.update(dict(current_state or {}))
        transition_signal = _context_shift_signal(
            current_state=current_state,
            relational_world=world_snapshot,
            identity_trace=identity_trace,
            community_profile_trace=community_profile_trace,
            sensor_input=current_state,
        )
        if transition_signal["transition_intensity"] > 0.0:
            merged_state["recent_strain"] = _float_from(merged_state, "recent_strain", default=0.32) + transition_signal["transition_intensity"] * 0.12
            merged_state["community_resonance"] = max(0.0, _float_from(merged_state, "community_resonance", default=0.0) - transition_signal["transition_intensity"] * 0.12)
            merged_state["culture_resonance"] = max(0.0, _float_from(merged_state, "culture_resonance", default=0.0) - transition_signal["transition_intensity"] * 0.08)
        environment_pressure = self.environment_pressure_core.snapshot(
            relational_world=world_snapshot,
            sensor_input=current_state,
            current_state=merged_state,
        )
        relationship = self.relationship_core.post_turn(
            previous=current_state,
            relational_world=world_snapshot,
            reply_present=bool(reply_text),
            stress=stress,
            recovery_need=recovery_need,
        )
        development = self.development_core.post_turn(
            previous=current_state,
            relational_world=world_snapshot,
            reply_present=bool(reply_text),
            stress=stress,
            recovery_need=recovery_need,
            environment_pressure=environment_pressure.to_dict(),
            terrain_transition_roughness=_float_from(current_state, "terrain_transition_roughness", default=0.0),
            recalled_tentative_bias=_float_from(current_state, "recalled_tentative_bias", default=0.0),
            recovery_reopening=_float_from(current_state, "recovery_reopening", default=0.0),
        )
        if transferred_lessons:
            development = self.development_core.apply_transferred_learning(
                previous=development.to_dict(),
                lessons=transferred_lessons,
            )
        personality = self.personality_core.snapshot(
            current_state=merged_state,
            development=development.to_dict(),
            relationship=relationship.to_dict(),
            environment_pressure=environment_pressure.to_dict(),
        )
        audit_expression_hints = _expression_hints(
            terrain_transition_roughness=_float_from(current_state, "terrain_transition_roughness", default=0.0),
            caution_bias=personality.caution_bias,
            recent_strain=_float_from(merged_state, "recent_strain", default=0.32),
            continuity_score=_float_from(merged_state, "continuity_score", default=0.48),
        )
        merged_state["terrain_transition_roughness"] = _float_from(current_state, "terrain_transition_roughness", default=0.0)
        reconstructed = self.reinterpretation_core.build_reconstructed_record(
            recall_payload=recall_payload or {},
            current_state=merged_state,
            relational_world=world_snapshot,
            environment_pressure=environment_pressure.to_dict(),
            transition_signal=transition_signal,
            reply_text=reply_text,
            user_text=str(user_input.get("text") or "").strip(),
        )
        persistence = self.persistence_core.post_turn(
            previous=current_state,
            development=development.to_dict(),
            relationship=relationship.to_dict(),
            personality=personality.to_dict(),
            environment_pressure=environment_pressure.to_dict(),
            transition_signal=transition_signal,
            reply_present=bool(reply_text),
            reconstructed_memory_appended=reconstructed is not None,
            transferred_lessons_used=len(transferred_lessons or []),
        )
        future_signal = _float_from(current_state, "future_signal", default=0.0)
        interaction_afterglow = _interaction_afterglow_from_surface_policy(
            surface_policy_active=_float_from(current_state, "surface_policy_active", default=0.0),
            surface_policy_intent=_text_or_none(current_state.get("surface_policy_intent")),
            social_update_strength=development.social_update_strength,
            identity_update_strength=development.identity_update_strength,
            terrain_transition_roughness=_float_from(current_state, "terrain_transition_roughness", default=0.0),
        )
        forgetting_snapshot = {
            "forgetting_pressure": _float_from(current_state, "forgetting_pressure", default=0.0),
            "replay_horizon": max(1, int(_float_from(current_state, "replay_horizon", default=2.0))),
        }
        memory_orchestration = self.memory_orchestration_core.snapshot(
            relational_world=world_snapshot,
            current_state=current_state,
            forgetting_snapshot=forgetting_snapshot,
            recall_active=bool(recall_payload),
        ).to_dict()
        raw_core_axes = _core_state_axes(
            stress=lingering,
            recovery_need=max(0.0, recovery_need * 0.92),
            temporal_pressure=temporal_pressure,
            continuity_score=persistence.continuity_score,
            social_grounding=persistence.social_grounding,
            recent_strain=persistence.recent_strain,
            trust_bias=development.trust_bias,
            terrain_transition_roughness=_float_from(current_state, "terrain_transition_roughness", default=0.0),
            interaction_afterglow=interaction_afterglow,
            transition_intensity=_float_from(transition_signal, "transition_intensity", default=0.0),
            community_resonance=persistence.community_resonance,
            culture_resonance=persistence.culture_resonance,
            future_signal=future_signal,
            recall_active=bool(recall_payload),
        )
        core_axes = _evolve_core_axes(
            previous=current_state,
            current=raw_core_axes,
            stress=lingering,
            recovery_need=max(0.0, recovery_need * 0.92),
            continuity_score=persistence.continuity_score,
            social_grounding=persistence.social_grounding,
            recall_active=bool(recall_payload),
            interaction_afterglow=interaction_afterglow,
        )
        next_state = HookState(
            stress=lingering,
            recovery_need=max(0.0, recovery_need * 0.92),
            attention_density=_float_from(current_state, "attention_density", default=0.0),
            safety_bias=_float_from(current_state, "safety_bias", default=0.0),
            temporal_pressure=temporal_pressure,
            memory_anchor=_text_or_none(current_state.get("memory_anchor")),
            replay_active=bool(current_state.get("replay_active", False)),
            route=str(current_state.get("route") or "reflex"),
            talk_mode=str(current_state.get("talk_mode") or "watch"),
            body_state_flag=str(current_state.get("body_state_flag") or "normal"),
            voice_level=_float_from(current_state, "voice_level", default=0.0),
            person_count=int(current_state.get("person_count", 0) or 0),
            autonomic_balance=_float_from(current_state, "autonomic_balance", default=0.5),
            belonging=development.belonging,
            recalled_tentative_bias=_float_from(current_state, "recalled_tentative_bias", default=0.0),
            social_update_strength=development.social_update_strength,
            identity_update_strength=development.identity_update_strength,
            interaction_afterglow=interaction_afterglow,
            interaction_afterglow_intent=_text_or_none(current_state.get("surface_policy_intent")),
            replay_intensity=core_axes["replay_intensity"],
            anticipation_tension=core_axes["anticipation_tension"],
            stabilization_drive=core_axes["stabilization_drive"],
            relational_clarity=core_axes["relational_clarity"],
            meaning_inertia=core_axes["meaning_inertia"],
            recovery_reopening=core_axes["recovery_reopening"],
            trust_bias=development.trust_bias,
            norm_pressure=development.norm_pressure,
            role_commitment=development.role_commitment,
            attachment=relationship.attachment,
            trust_memory=relationship.trust_memory,
            familiarity=relationship.familiarity,
            role_alignment=relationship.role_alignment,
            rupture_sensitivity=relationship.rupture_sensitivity,
            caution_bias=personality.caution_bias,
            affiliation_bias=personality.affiliation_bias,
            exploration_bias=personality.exploration_bias,
            reflective_bias=personality.reflective_bias,
            continuity_score=persistence.continuity_score,
            social_grounding=persistence.social_grounding,
            recent_strain=persistence.recent_strain,
            culture_resonance=persistence.culture_resonance,
            community_resonance=persistence.community_resonance,
            terrain_transition_roughness=_float_from(current_state, "terrain_transition_roughness", default=0.0),
            reuse_trajectory=memory_orchestration["reuse_trajectory"],
            interference_pressure=memory_orchestration["interference_pressure"],
            consolidation_priority=memory_orchestration["consolidation_priority"],
            prospective_memory_pull=memory_orchestration["prospective_memory_pull"],
        )
        appends = [dict(item) for item in (memory_write_candidates or [])]
        if reconstructed:
            appends.append(reconstructed)
        lesson_appends = [dict(item) for item in (transferred_lessons or []) if isinstance(item, Mapping)]
        if lesson_appends:
            appends.extend(lesson_appends)
        appends.append(
            {
                "kind": "relationship_trace",
                "summary": "slow relationship trace",
                "text": "slow relationship trace",
                "memory_anchor": self.relational_world_core.snapshot().get("place_memory_anchor") or _text_or_none(current_state.get("memory_anchor")) or "relationship-trace",
                "culture_id": world_snapshot.get("culture_id"),
                "community_id": world_snapshot.get("community_id"),
                "social_role": world_snapshot.get("social_role"),
                "profile_scope": "community_role_place",
                "attachment": round(next_state.attachment, 4),
                "trust_memory": round(next_state.trust_memory, 4),
                "familiarity": round(next_state.familiarity, 4),
                "role_alignment": round(next_state.role_alignment, 4),
                "rupture_sensitivity": round(next_state.rupture_sensitivity, 4),
                "caution_bias": round(next_state.caution_bias, 4),
                "affiliation_bias": round(next_state.affiliation_bias, 4),
                "reuse_trajectory": round(next_state.reuse_trajectory, 4),
                "consolidation_priority": round(next_state.consolidation_priority, 4),
            }
        )
        appends.append(
            {
                "kind": "identity_trace",
                "summary": "slow identity trace",
                "text": "slow identity trace",
                "memory_anchor": _text_or_none(current_state.get("memory_anchor")) or "identity-trace",
                "culture_id": world_snapshot.get("culture_id"),
                "community_id": world_snapshot.get("community_id"),
                "social_role": world_snapshot.get("social_role"),
                "continuity_score": round(next_state.continuity_score, 4),
                "social_grounding": round(next_state.social_grounding, 4),
                "recent_strain": round(next_state.recent_strain, 4),
                "caution_bias": round(next_state.caution_bias, 4),
                "affiliation_bias": round(next_state.affiliation_bias, 4),
                "belonging": round(next_state.belonging, 4),
                "trust_bias": round(next_state.trust_bias, 4),
                "attachment": round(next_state.attachment, 4),
                "familiarity": round(next_state.familiarity, 4),
                "social_update_strength": round(development.social_update_strength, 4),
                "identity_update_strength": round(development.identity_update_strength, 4),
                "prospective_memory_pull": round(next_state.prospective_memory_pull, 4),
                "interference_pressure": round(next_state.interference_pressure, 4),
            }
        )
        appends.append(
            {
                "kind": "community_profile_trace",
                "summary": "slow community profile",
                "text": "slow community profile",
                "memory_anchor": world_snapshot.get("community_id") or world_snapshot.get("culture_id") or "community-profile",
                "culture_id": world_snapshot.get("culture_id"),
                "community_id": world_snapshot.get("community_id"),
                "social_role": world_snapshot.get("social_role"),
                "profile_scope": "culture_community",
                "belonging": round(next_state.belonging, 4),
                "trust_bias": round(next_state.trust_bias, 4),
                "norm_pressure": round(next_state.norm_pressure, 4),
                "role_commitment": round(next_state.role_commitment, 4),
                "culture_resonance": round(next_state.culture_resonance, 4),
                "community_resonance": round(next_state.community_resonance, 4),
                "ritual_memory": round(environment_pressure.ritual_pressure, 4),
                "institutional_memory": round(environment_pressure.institutional_pressure, 4),
                "social_update_strength": round(development.social_update_strength, 4),
                "identity_update_strength": round(development.identity_update_strength, 4),
                "prospective_memory_pull": round(next_state.prospective_memory_pull, 4),
                "interference_pressure": round(next_state.interference_pressure, 4),
            }
        )
        if reply_text:
            appends.append(
                {
                    "kind": "observed",
                    "source": "post_turn",
                    "user_text": str(user_input.get("text") or "").strip(),
                    "assistant_text": reply_text,
                    "memory_anchor": _text_or_none(current_state.get("memory_anchor")),
                    "summary": reply_text,
                    "text": reply_text,
                    "surface_policy_active": _float_from(current_state, "surface_policy_active", default=0.0),
                    "surface_policy_level": _text_or_none(current_state.get("surface_policy_level")),
                    "surface_policy_intent": _text_or_none(current_state.get("surface_policy_intent")),
                    "consolidation_priority": round(next_state.consolidation_priority, 4),
                    "prospective_memory_pull": round(next_state.prospective_memory_pull, 4),
                }
            )
        stored = self.memory_core.append_records(appends)
        return PostTurnUpdateResult(
            state=next_state,
            memory_appends=stored,
            audit_record={
                "kind": "thin_audit",
                "route": next_state.route,
                "talk_mode": next_state.talk_mode,
                "reply_present": bool(reply_text),
                "temporal_pressure": round(next_state.temporal_pressure, 4),
                "culture_id": world_snapshot.get("culture_id"),
                "community_id": world_snapshot.get("community_id"),
                "belonging": round(next_state.belonging, 4),
                "trust_bias": round(next_state.trust_bias, 4),
                "norm_pressure": round(next_state.norm_pressure, 4),
                "role_commitment": round(next_state.role_commitment, 4),
                "attachment": round(next_state.attachment, 4),
                "trust_memory": round(next_state.trust_memory, 4),
                "familiarity": round(next_state.familiarity, 4),
                "caution_bias": round(next_state.caution_bias, 4),
                "affiliation_bias": round(next_state.affiliation_bias, 4),
                "reflective_bias": round(next_state.reflective_bias, 4),
                "continuity_score": round(next_state.continuity_score, 4),
                "social_grounding": round(next_state.social_grounding, 4),
                "recent_strain": round(next_state.recent_strain, 4),
                "culture_resonance": round(next_state.culture_resonance, 4),
                "community_resonance": round(next_state.community_resonance, 4),
                "transferred_lessons_used": len(transferred_lessons or []),
                "reconstructed_memory_appended": reconstructed is not None,
                "reinterpretation_mode": (reconstructed or {}).get("reinterpretation_mode"),
                "tentative_bias": audit_expression_hints.get("tentative_bias"),
                "assertiveness_cap": audit_expression_hints.get("assertiveness_cap"),
                "social_update_strength": round(development.social_update_strength, 4),
                "identity_update_strength": round(development.identity_update_strength, 4),
                "interaction_afterglow": round(interaction_afterglow, 4),
                "interaction_afterglow_intent": _text_or_none(current_state.get("surface_policy_intent")),
                "replay_intensity": round(core_axes["replay_intensity"], 4),
                "anticipation_tension": round(core_axes["anticipation_tension"], 4),
                "stabilization_drive": round(core_axes["stabilization_drive"], 4),
                "relational_clarity": round(core_axes["relational_clarity"], 4),
                "meaning_inertia": round(core_axes["meaning_inertia"], 4),
                "recovery_reopening": round(core_axes["recovery_reopening"], 4),
                "reuse_trajectory": round(memory_orchestration["reuse_trajectory"], 4),
                "interference_pressure": round(memory_orchestration["interference_pressure"], 4),
                "consolidation_priority": round(memory_orchestration["consolidation_priority"], 4),
                "prospective_memory_pull": round(memory_orchestration["prospective_memory_pull"], 4),
                "memory_orchestration": dict(memory_orchestration),
                "environment_pressure": environment_pressure.to_dict(),
            },
        )


def _context_shift_signal(
    *,
    current_state: Optional[Mapping[str, Any]],
    relational_world: Mapping[str, Any],
    identity_trace: Optional[Mapping[str, Any]],
    community_profile_trace: Optional[Mapping[str, Any]],
    sensor_input: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    previous_culture = _text_or_none((current_state or {}).get("culture_id")) or _text_or_none((identity_trace or {}).get("culture_id"))
    previous_community = _text_or_none((current_state or {}).get("community_id")) or _text_or_none((community_profile_trace or {}).get("community_id"))
    previous_role = _text_or_none((current_state or {}).get("social_role")) or _text_or_none((identity_trace or {}).get("social_role"))
    previous_place = _text_or_none((current_state or {}).get("memory_anchor")) or _text_or_none((identity_trace or {}).get("memory_anchor"))
    previous_body_flag = _text_or_none((current_state or {}).get("body_state_flag"))
    previous_person_count = int((current_state or {}).get("person_count", 0) or 0)
    current_culture = _text_or_none(relational_world.get("culture_id"))
    current_community = _text_or_none(relational_world.get("community_id"))
    current_role = _text_or_none(relational_world.get("social_role"))
    current_place = _text_or_none(relational_world.get("place_memory_anchor"))
    current_body_flag = _text_or_none((sensor_input or {}).get("body_state_flag")) or previous_body_flag
    current_person_count = int((sensor_input or {}).get("person_count", previous_person_count) or 0)
    privacy_tags = [str(tag).lower() for tag in ((sensor_input or {}).get("privacy_tags") or [])]
    culture_changed = bool(previous_culture and current_culture and previous_culture != current_culture)
    community_changed = bool(previous_community and current_community and previous_community != current_community)
    role_changed = bool(previous_role and current_role and previous_role != current_role)
    place_changed = bool(previous_place and current_place and previous_place != current_place)
    body_state_changed = bool(previous_body_flag and current_body_flag and previous_body_flag != current_body_flag)
    privacy_shift = "private" in privacy_tags and previous_body_flag != "private_high_arousal"
    density_shift = abs(current_person_count - previous_person_count) >= 2
    social_discontinuity = _clamp01(
        (CONTEXT_SHIFT_WEIGHTS["culture_change"] if culture_changed else 0.0)
        + (CONTEXT_SHIFT_WEIGHTS["community_change"] if community_changed else 0.0)
        + (CONTEXT_SHIFT_WEIGHTS["role_change"] if role_changed else 0.0)
    )
    situational_discontinuity = _clamp01(
        (CONTEXT_SHIFT_WEIGHTS["place_change"] if place_changed else 0.0)
        + (CONTEXT_SHIFT_WEIGHTS["privacy_shift"] if privacy_shift else 0.0)
        + (CONTEXT_SHIFT_WEIGHTS["density_shift"] if density_shift else 0.0)
    )
    bodily_discontinuity = _clamp01(
        (CONTEXT_SHIFT_WEIGHTS["body_change"] if body_state_changed else 0.0)
    )
    transition_intensity = _clamp01(
        social_discontinuity * CONTEXT_SHIFT_WEIGHTS["social_integration"]
        + situational_discontinuity * CONTEXT_SHIFT_WEIGHTS["situational_integration"]
        + bodily_discontinuity * CONTEXT_SHIFT_WEIGHTS["bodily_integration"]
    )
    return {
        "culture_changed": culture_changed,
        "community_changed": community_changed,
        "role_changed": role_changed,
        "place_changed": place_changed,
        "body_state_changed": body_state_changed,
        "privacy_shift": privacy_shift,
        "density_shift": density_shift,
        "social_discontinuity": round(social_discontinuity, 4),
        "situational_discontinuity": round(situational_discontinuity, 4),
        "bodily_discontinuity": round(bodily_discontinuity, 4),
        "transition_intensity": round(transition_intensity, 4),
    }


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _mapping_or_none(value: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return value
    return None


def _expression_hints(*, terrain_transition_roughness: float, caution_bias: float, recent_strain: float, continuity_score: float, social_update_strength: float = 1.0, identity_update_strength: float = 1.0, interaction_afterglow: float = 0.0, interaction_afterglow_intent: Optional[str] = None, replay_intensity: float = 0.0, anticipation_tension: float = 0.0, stabilization_drive: float = 0.0, relational_clarity: float = 0.0, meaning_inertia: float = 0.0, recovery_reopening: float = 0.0, object_affordance_bias: float = 0.0, fragility_guard: float = 0.0, object_attachment: float = 0.0, object_avoidance: float = 0.0, tool_extension_bias: float = 0.0, ritually_sensitive_bias: float = 0.0, defensive_salience: float = 0.0, reachability: float = 0.0) -> Dict[str, Any]:
    tentative_bias = _clamp01(
        terrain_transition_roughness * EXPRESSION_HINT_WEIGHTS["tentative_roughness"]
        + caution_bias * EXPRESSION_HINT_WEIGHTS["tentative_caution"]
        + recent_strain * EXPRESSION_HINT_WEIGHTS["tentative_recent_strain"]
        + anticipation_tension * EXPRESSION_HINT_WEIGHTS["tentative_anticipation"]
        + meaning_inertia * EXPRESSION_HINT_WEIGHTS["tentative_inertia"]
        + fragility_guard * 0.08
        + object_avoidance * 0.08
        + defensive_salience * 0.06
        - continuity_score * EXPRESSION_HINT_WEIGHTS["tentative_continuity_relief"]
        - relational_clarity * EXPRESSION_HINT_WEIGHTS["tentative_clarity_relief"]
    )
    assertiveness_cap = max(0.2, 1.0 - terrain_transition_roughness * EXPRESSION_HINT_WEIGHTS["assertive_roughness"] - caution_bias * EXPRESSION_HINT_WEIGHTS["assertive_caution"] - anticipation_tension * EXPRESSION_HINT_WEIGHTS["assertive_anticipation"] - fragility_guard * 0.08 - object_avoidance * 0.06 - defensive_salience * 0.06 + reachability * 0.02)
    clarify_first = identity_update_strength <= 0.68 or social_update_strength <= 0.72 or interaction_afterglow >= 0.24 or stabilization_drive >= 0.42 or anticipation_tension >= 0.4
    return {
        "meaning_pacing": "slow" if tentative_bias >= 0.32 or meaning_inertia >= 0.4 else "steady",
        "tentative_bias": round(tentative_bias, 4),
        "assertiveness_cap": round(assertiveness_cap, 4),
        "avoid_definitive_interpretation": terrain_transition_roughness >= 0.28 or identity_update_strength <= 0.66 or anticipation_tension >= 0.42 or meaning_inertia >= 0.44,
        "favor_grounded_observation": terrain_transition_roughness >= 0.2 or stabilization_drive >= 0.38,
        "clarify_first": clarify_first,
        "question_bias": round(_clamp01((1.0 - identity_update_strength) * EXPRESSION_HINT_WEIGHTS["question_identity_gap"] + (1.0 - social_update_strength) * EXPRESSION_HINT_WEIGHTS["question_social_gap"] + interaction_afterglow * EXPRESSION_HINT_WEIGHTS["question_afterglow"] + anticipation_tension * EXPRESSION_HINT_WEIGHTS["question_anticipation"] + stabilization_drive * EXPRESSION_HINT_WEIGHTS["question_stabilization"]), 4),
        "interaction_pacing": "check" if clarify_first else "flow",
        "carry_gentleness": interaction_afterglow >= 0.24 or stabilization_drive >= 0.42,
        "allow_reopening": recovery_reopening >= 0.2 and meaning_inertia <= 0.52,
        "handle_gently": fragility_guard >= 0.24 or ritually_sensitive_bias >= 0.22,
        "object_affordance_bias": round(object_affordance_bias, 4),
        "fragility_guard": round(fragility_guard, 4),
        "object_attachment": round(object_attachment, 4),
        "object_avoidance": round(object_avoidance, 4),
        "tool_extension_bias": round(tool_extension_bias, 4),
        "defensive_salience": round(defensive_salience, 4),
        "reachability": round(reachability, 4),
        "interaction_afterglow_intent": interaction_afterglow_intent,
        "replay_intensity": round(replay_intensity, 4),
        "anticipation_tension": round(anticipation_tension, 4),
        "stabilization_drive": round(stabilization_drive, 4),
        "relational_clarity": round(relational_clarity, 4),
        "meaning_inertia": round(meaning_inertia, 4),
        "recovery_reopening": round(recovery_reopening, 4),
    }

def _core_state_axes(
    *,
    stress: float,
    recovery_need: float,
    temporal_pressure: float,
    continuity_score: float,
    social_grounding: float,
    recent_strain: float,
    trust_bias: float,
    terrain_transition_roughness: float,
    interaction_afterglow: float,
    transition_intensity: float,
    community_resonance: float,
    culture_resonance: float,
    future_signal: float = 0.0,
    recall_active: bool = False,
) -> Dict[str, float]:
    replay_intensity = _clamp01(interaction_afterglow * CORE_AXIS_WEIGHTS["replay_afterglow"] + terrain_transition_roughness * CORE_AXIS_WEIGHTS["replay_roughness"] + recent_strain * CORE_AXIS_WEIGHTS["replay_recent_strain"] + (1.0 - continuity_score) * CORE_AXIS_WEIGHTS["replay_discontinuity"] + (CORE_AXIS_WEIGHTS["replay_recall_active"] if recall_active else 0.0))
    anticipation_tension = _clamp01(temporal_pressure * CORE_AXIS_WEIGHTS["anticipation_temporal"] + stress * CORE_AXIS_WEIGHTS["anticipation_stress"] + recovery_need * CORE_AXIS_WEIGHTS["anticipation_recovery_need"] + future_signal * CORE_AXIS_WEIGHTS["anticipation_future_signal"] + terrain_transition_roughness * CORE_AXIS_WEIGHTS["anticipation_roughness"] + recent_strain * CORE_AXIS_WEIGHTS["anticipation_recent_strain"] - continuity_score * CORE_AXIS_WEIGHTS["anticipation_continuity_relief"])
    stabilization_drive = _clamp01(recovery_need * CORE_AXIS_WEIGHTS["stabilization_recovery_need"] + terrain_transition_roughness * CORE_AXIS_WEIGHTS["stabilization_roughness"] + anticipation_tension * CORE_AXIS_WEIGHTS["stabilization_anticipation"] + interaction_afterglow * CORE_AXIS_WEIGHTS["stabilization_afterglow"] + transition_intensity * CORE_AXIS_WEIGHTS["stabilization_transition"] - social_grounding * CORE_AXIS_WEIGHTS["stabilization_social_grounding_relief"])
    relational_clarity = _clamp01(trust_bias * CORE_AXIS_WEIGHTS["clarity_trust"] + continuity_score * CORE_AXIS_WEIGHTS["clarity_continuity"] + social_grounding * CORE_AXIS_WEIGHTS["clarity_social_grounding"] + community_resonance * CORE_AXIS_WEIGHTS["clarity_community_resonance"] + culture_resonance * CORE_AXIS_WEIGHTS["clarity_culture_resonance"] - terrain_transition_roughness * CORE_AXIS_WEIGHTS["clarity_roughness_penalty"] - recent_strain * CORE_AXIS_WEIGHTS["clarity_recent_strain_penalty"] - transition_intensity * CORE_AXIS_WEIGHTS["clarity_transition_penalty"])
    meaning_inertia = _clamp01(terrain_transition_roughness * CORE_AXIS_WEIGHTS["inertia_roughness"] + interaction_afterglow * CORE_AXIS_WEIGHTS["inertia_afterglow"] + anticipation_tension * CORE_AXIS_WEIGHTS["inertia_anticipation"] + recent_strain * CORE_AXIS_WEIGHTS["inertia_recent_strain"] + (1.0 - relational_clarity) * CORE_AXIS_WEIGHTS["inertia_clarity_gap"] - continuity_score * CORE_AXIS_WEIGHTS["inertia_continuity_relief"])
    return {
        "replay_intensity": round(replay_intensity, 4),
        "anticipation_tension": round(anticipation_tension, 4),
        "stabilization_drive": round(stabilization_drive, 4),
        "relational_clarity": round(relational_clarity, 4),
        "meaning_inertia": round(meaning_inertia, 4),
    }

def _evolve_core_axes(
    *,
    previous: Optional[Mapping[str, Any]],
    current: Mapping[str, float],
    stress: float,
    recovery_need: float,
    continuity_score: float,
    social_grounding: float,
    recall_active: bool,
    interaction_afterglow: float,
) -> Dict[str, float]:
    replay_prev = _float_from(previous, "replay_intensity", default=0.0)
    anticipation_prev = _float_from(previous, "anticipation_tension", default=0.0)
    stabilization_prev = _float_from(previous, "stabilization_drive", default=0.0)
    clarity_prev = _float_from(previous, "relational_clarity", default=0.0)
    inertia_prev = _float_from(previous, "meaning_inertia", default=0.0)

    decay = _clamp01(CORE_EVOLUTION_WEIGHTS["decay_base"] + social_grounding * CORE_EVOLUTION_WEIGHTS["decay_social_grounding"] + continuity_score * CORE_EVOLUTION_WEIGHTS["decay_continuity"] - stress * CORE_EVOLUTION_WEIGHTS["decay_stress_penalty"])
    reinforcement = _clamp01((CORE_EVOLUTION_WEIGHTS["reinforcement_recall"] if recall_active else 0.0) + interaction_afterglow * CORE_EVOLUTION_WEIGHTS["reinforcement_afterglow"] + stress * CORE_EVOLUTION_WEIGHTS["reinforcement_stress"])
    recovery_reopening = _clamp01(continuity_score * CORE_EVOLUTION_WEIGHTS["reopening_continuity"] + social_grounding * CORE_EVOLUTION_WEIGHTS["reopening_social_grounding"] - recovery_need * CORE_EVOLUTION_WEIGHTS["reopening_recovery_penalty"] - stress * CORE_EVOLUTION_WEIGHTS["reopening_stress_penalty"])

    replay_intensity = _clamp01(current.get("replay_intensity", 0.0) * (CORE_EVOLUTION_WEIGHTS["replay_current"] + reinforcement * CORE_EVOLUTION_WEIGHTS["replay_reinforcement"]) + replay_prev * (CORE_EVOLUTION_WEIGHTS["replay_previous"] + decay * CORE_EVOLUTION_WEIGHTS["replay_decay"]) - recovery_reopening * CORE_EVOLUTION_WEIGHTS["replay_reopening_relief"])
    anticipation_tension = _clamp01(current.get("anticipation_tension", 0.0) * CORE_EVOLUTION_WEIGHTS["anticipation_current"] + anticipation_prev * (CORE_EVOLUTION_WEIGHTS["anticipation_previous"] + decay * CORE_EVOLUTION_WEIGHTS["anticipation_decay"]) - recovery_reopening * CORE_EVOLUTION_WEIGHTS["anticipation_reopening_relief"])
    stabilization_drive = _clamp01(current.get("stabilization_drive", 0.0) * CORE_EVOLUTION_WEIGHTS["stabilization_current"] + stabilization_prev * (CORE_EVOLUTION_WEIGHTS["stabilization_previous"] + decay * CORE_EVOLUTION_WEIGHTS["stabilization_decay"]) + recovery_need * CORE_EVOLUTION_WEIGHTS["stabilization_recovery"])
    relational_clarity = _clamp01(current.get("relational_clarity", 0.0) * CORE_EVOLUTION_WEIGHTS["clarity_current"] + clarity_prev * (CORE_EVOLUTION_WEIGHTS["clarity_previous"] + decay * CORE_EVOLUTION_WEIGHTS["clarity_decay"]) + recovery_reopening * CORE_EVOLUTION_WEIGHTS["clarity_reopening"])
    meaning_inertia = _clamp01(current.get("meaning_inertia", 0.0) * CORE_EVOLUTION_WEIGHTS["inertia_current"] + inertia_prev * (CORE_EVOLUTION_WEIGHTS["inertia_previous"] + decay * CORE_EVOLUTION_WEIGHTS["inertia_decay"]) - recovery_reopening * CORE_EVOLUTION_WEIGHTS["inertia_reopening_relief"])

    return {
        "replay_intensity": round(replay_intensity, 4),
        "anticipation_tension": round(anticipation_tension, 4),
        "stabilization_drive": round(stabilization_drive, 4),
        "relational_clarity": round(relational_clarity, 4),
        "meaning_inertia": round(meaning_inertia, 4),
        "decay_factor": round(decay, 4),
        "reinforcement_factor": round(reinforcement, 4),
        "recovery_reopening": round(recovery_reopening, 4),
    }

def _interaction_afterglow_from_surface_policy(
    *,
    surface_policy_active: float,
    surface_policy_intent: Optional[str],
    social_update_strength: float,
    identity_update_strength: float,
    terrain_transition_roughness: float,
) -> float:
    base = _clamp01(surface_policy_active) * AFTERGLOW_WEIGHTS["surface_active"]
    if surface_policy_intent == "clarify":
        base += AFTERGLOW_WEIGHTS["clarify_bonus"]
    elif surface_policy_intent == "check_in":
        base += AFTERGLOW_WEIGHTS["check_in_bonus"]
    base += (1.0 - _clamp01(social_update_strength)) * AFTERGLOW_WEIGHTS["social_gap"]
    base += (1.0 - _clamp01(identity_update_strength)) * AFTERGLOW_WEIGHTS["identity_gap"]
    base += _clamp01(terrain_transition_roughness) * AFTERGLOW_WEIGHTS["roughness"]
    return _clamp01(base)

def _response_expression_allocation(
    *,
    stress: float,
    recovery_need: float,
    safety_bias: float,
    body_state_flag: str,
    autonomic_balance: float,
    norm_pressure: float,
    trust_bias: float,
    caution_bias: float,
    affiliation_bias: float,
    continuity_score: float,
    social_grounding: float,
    recent_strain: float,
    culture_resonance: float,
    community_resonance: float,
    terrain_transition_roughness: float,
    recalled_tentative_bias: float,
    anticipation_tension: float,
    stabilization_drive: float,
    relational_clarity: float,
    meaning_inertia: float,
) -> Dict[str, float]:
    hold_back = max(
        stress * RESPONSE_GATE_WEIGHTS["hesitation_stress"],
        recovery_need * RESPONSE_GATE_WEIGHTS["hesitation_recovery_need"],
        safety_bias,
        norm_pressure * RESPONSE_GATE_WEIGHTS["hesitation_norm"],
        caution_bias * RESPONSE_GATE_WEIGHTS["hesitation_caution"],
        recent_strain * RESPONSE_GATE_WEIGHTS["hesitation_recent_strain"],
        terrain_transition_roughness * RESPONSE_GATE_WEIGHTS["hesitation_roughness"],
        recalled_tentative_bias * RESPONSE_GATE_WEIGHTS["hesitation_recalled_tentative"],
        anticipation_tension * RESPONSE_GATE_WEIGHTS["hesitation_anticipation"],
        stabilization_drive * RESPONSE_GATE_WEIGHTS["hesitation_stabilization"],
        meaning_inertia * RESPONSE_GATE_WEIGHTS["hesitation_inertia"],
    )
    hold_relief = min(
        RESPONSE_ALLOCATION_WEIGHTS["hold_relief_sum_cap"],
        trust_bias * RESPONSE_GATE_WEIGHTS["hesitation_trust_relief"]
        + continuity_score * RESPONSE_GATE_WEIGHTS["hesitation_continuity_relief"]
        + social_grounding * RESPONSE_GATE_WEIGHTS["hesitation_social_grounding_relief"]
        + community_resonance * RESPONSE_GATE_WEIGHTS["hesitation_community_relief"]
        + relational_clarity * RESPONSE_GATE_WEIGHTS["hesitation_clarity_relief"],
    )
    hold_back = _clamp01(hold_back - hold_relief)
    if body_state_flag == "private_high_arousal":
        hold_back = max(hold_back, RESPONSE_GATE_WEIGHTS["private_high_arousal_floor"])
    if autonomic_balance < RESPONSE_GATE_WEIGHTS["autonomic_floor_threshold"]:
        hold_back = max(hold_back, RESPONSE_GATE_WEIGHTS["autonomic_floor_value"])
    express_support = min(
        RESPONSE_ALLOCATION_WEIGHTS["express_support_cap"],
        affiliation_bias * RESPONSE_GATE_WEIGHTS["surface_affiliation"]
        + culture_resonance * RESPONSE_GATE_WEIGHTS["surface_culture"]
        + community_resonance * RESPONSE_GATE_WEIGHTS["surface_community"]
        + relational_clarity * RESPONSE_GATE_WEIGHTS["surface_clarity"],
    )
    express_penalty = min(
        RESPONSE_ALLOCATION_WEIGHTS["express_penalty_cap"],
        terrain_transition_roughness * RESPONSE_GATE_WEIGHTS["surface_roughness_penalty"]
        + recalled_tentative_bias * RESPONSE_GATE_WEIGHTS["surface_recalled_tentative_penalty"]
        + anticipation_tension * RESPONSE_GATE_WEIGHTS["surface_anticipation_penalty"],
    )
    express_now = _clamp01((1.0 - hold_back) * (RESPONSE_ALLOCATION_WEIGHTS["express_base_scale"] + express_support - express_penalty))
    hold_back = round(_clamp01(1.0 - express_now), 4)
    express_now = round(express_now, 4)
    return {
        "express_now": express_now,
        "hold_back": hold_back,
    }

def _float_from(mapping: Optional[Mapping[str, Any]], key: str, *, default: float) -> float:
    if not isinstance(mapping, Mapping):
        return float(default)
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _text_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
