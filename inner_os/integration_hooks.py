from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, Mapping, Optional

import numpy as np

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
from .field_estimator_core import FieldEstimatorCore
from .interaction import (
    advance_interaction_stream,
    compose_nonverbal_profile,
    derive_live_interaction_regulation,
    derive_relational_mood,
    orchestrate_interaction,
    summarize_interaction_trace,
)
from .interaction.models import SituationState
from .interaction_option_search import generate_interaction_option_candidates
from .action_posture import derive_action_posture
from .actuation_plan import derive_actuation_plan
from .access_dynamics import advance_access_dynamics
from .access_projection import project_access_regions
from .affect_blend import derive_affect_blend_state
from .association_graph import (
    BasicAssociationGraph,
    apply_association_reinforcement,
    coerce_association_graph_state,
)
from .affective_localizer import BasicAffectiveLocalizer
from .affective_position import AffectivePositionState, make_neutral_affective_position
from .affective_terrain import (
    AffectiveTerrainState,
    BasicAffectiveTerrain,
    TerrainReadout,
    make_neutral_affective_terrain_state,
)
from .conscious_workspace import ignite_conscious_workspace
from .conversation_contract import build_conversation_contract
from .conversational_objects import derive_conversational_objects
from .contact_field import derive_contact_field
from .contact_dynamics import advance_contact_dynamics
from .contact_reflection_state import derive_contact_reflection_state
from .constraint_field import derive_constraint_field
from .dot_seed import derive_dot_seeds
from .group_thread_registry import update_group_thread_registry_snapshot as advance_group_thread_registry_snapshot
from .green_kernel_contracts import build_green_kernel_composition
from .interaction_effects import derive_interaction_effects
from .interaction_audit_bundle import build_interaction_audit_bundle
from .interaction_audit_casebook import (
    build_interaction_audit_case_entry,
    select_same_utterance_audit_cases,
    update_interaction_audit_casebook,
)
from .interaction_audit_report import build_interaction_audit_report
from .interaction_condition_report import build_interaction_condition_report
from .interaction_inspection_report import build_interaction_inspection_report
from .interaction_judgement_summary import derive_interaction_judgement_summary
from .interaction_judgement_view import derive_interaction_judgement_view
from .issue_state import derive_issue_state
from .policy_packet import derive_interaction_policy_packet
from .discussion_thread_state import derive_discussion_thread_state
from .object_operations import derive_object_operations
from .qualia_membrane_operator import derive_qualia_membrane_temporal_bias
from .qualia_kernel_adapter import RuntimeQualiaKernelAdapter
from .recent_dialogue_state import derive_recent_dialogue_state
from .resonance_evaluator import (
    evaluate_interaction_resonance,
    rerank_interaction_option_candidates,
)
from .relational_world import RelationalWorldCore
from .scene_state import derive_scene_state
from .terrain_core import AffectiveTerrainCore
from .working_memory_core import WorkingMemoryCore
from .partner_style import resolve_partner_utterance_stance
from .protection_mode import derive_protection_mode
from .insight_event import BasicInsightDetector
from .insight_trace import derive_insight_trace
from .relation_competition import (
    collect_related_person_ids,
    derive_relation_competition_state,
    summarize_person_registry_snapshot,
)
from .terrain_plasticity import (
    TerrainPlasticityUpdate,
    apply_terrain_plasticity,
    derive_terrain_plasticity_update,
)
from .temperament_estimate import advance_temperament_traces, derive_temperament_estimate
from .temporal_memory_orchestration import build_temporal_memory_evidence_bundle
from .expression.hint_bridge import (
    build_expression_hints_from_gate_result,
    build_qualia_planner_view_hint,
)
from .expression.surface_profile import derive_surface_profile
from .self_model import PersonRegistry, PersonNode, update_person_registry


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
    "hesitation_long_term_relief": 0.06,
    "private_high_arousal_floor": 0.55,
    "autonomic_floor_threshold": 0.42,
    "autonomic_floor_value": 0.42,
    "surface_floor": 0.15,
    "surface_affiliation": 0.08,
    "surface_culture": 0.04,
    "surface_community": 0.06,
    "surface_clarity": 0.04,
    "surface_long_term_support": 0.04,
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
    roughness_level: float = 0.0
    roughness_velocity: float = 0.0
    roughness_momentum: float = 0.0
    roughness_dwell: float = 0.0
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
    defensive_level: float = 0.0
    defensive_velocity: float = 0.0
    defensive_momentum: float = 0.0
    defensive_dwell: float = 0.0
    approach_confidence: float = 0.0
    reuse_trajectory: float = 0.0
    interference_pressure: float = 0.0
    consolidation_priority: float = 0.0
    prospective_memory_pull: float = 0.0
    working_memory_pressure: float = 0.0
    unresolved_count: int = 0
    current_focus: str = "ambient"
    pending_meaning: float = 0.0
    related_person_id: str = ""
    related_person_ids: list[str] = field(default_factory=list)
    long_term_theme_focus: str = ""
    long_term_theme_anchor: str = ""
    long_term_theme_kind: str = ""
    long_term_theme_summary: str = ""
    long_term_theme_strength: float = 0.0
    relation_seed_summary: str = ""
    partner_address_hint: str = ""
    partner_timing_hint: str = ""
    partner_stance_hint: str = ""
    partner_social_interpretation: str = ""
    conscious_residue_focus: str = ""
    conscious_residue_anchor: str = ""
    conscious_residue_summary: str = ""
    conscious_residue_strength: float = 0.0
    autobiographical_thread_mode: str = "none"
    autobiographical_thread_anchor: str = ""
    autobiographical_thread_focus: str = ""
    autobiographical_thread_strength: float = 0.0
    autobiographical_thread_reasons: list[str] = field(default_factory=list)
    interaction_alignment_score: float = 0.0
    shared_attention_delta: float = 0.0
    distance_mismatch: float = 0.0
    hesitation_mismatch: float = 0.0
    opening_pace_mismatch: float = 0.0
    return_gaze_mismatch: float = 0.0

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
    memory_evidence_bundle: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recall_payload": dict(self.recall_payload),
            "retrieval_summary": dict(self.retrieval_summary),
            "ignition_hints": dict(self.ignition_hints),
            "memory_evidence_bundle": dict(self.memory_evidence_bundle),
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
    person_registry_snapshot: Dict[str, Any] = field(default_factory=dict)
    group_thread_registry_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "memory_appends": list(self.memory_appends),
            "audit_record": dict(self.audit_record),
            "person_registry_snapshot": dict(self.person_registry_snapshot),
            "group_thread_registry_snapshot": dict(self.group_thread_registry_snapshot),
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
        field_estimator_core: Optional[FieldEstimatorCore] = None,
        working_memory_core: Optional[WorkingMemoryCore] = None,
        runtime_qualia_adapter: Optional[RuntimeQualiaKernelAdapter] = None,
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
        self.field_estimator_core = field_estimator_core or FieldEstimatorCore()
        self.working_memory_core = working_memory_core or WorkingMemoryCore()
        self.runtime_qualia_adapter = runtime_qualia_adapter or RuntimeQualiaKernelAdapter()

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
        working_memory_seed = _mapping_or_empty(local_context.get("working_memory_seed"))
        registry_snapshot = _mapping_or_empty(local_context.get("person_registry"))
        explicit_counterpart_person_id = _text_or_none(
            world_context.get("person_id")
            or (current_state or {}).get("related_person_id")
            or ((current_state or {}).get("related_person_ids") or [None])[0]
            or local_context.get("person_id")
            or working_memory_seed.get("related_person_id")
        )
        related_person_ids = collect_related_person_ids(
            explicit_counterpart_person_id,
            (current_state or {}).get("related_person_ids"),
            working_memory_seed.get("related_person_ids"),
            registry_snapshot=registry_snapshot,
            limit=4,
        )
        counterpart_person_id = explicit_counterpart_person_id
        if not counterpart_person_id and related_person_ids:
            counterpart_person_id = derive_relation_competition_state(
                self_state=current_state or {},
                related_person_ids=related_person_ids,
                registry_snapshot=registry_snapshot,
                limit=4,
            ).dominant_person_id or None
        identity_trace = self.memory_core.load_latest_identity_trace(
            culture_id=str(world_context.get("culture_id") or "") or None,
            community_id=str(world_context.get("community_id") or "") or None,
            social_role=str(world_context.get("social_role") or "") or None,
            related_person_id=counterpart_person_id,
        )
        relationship_trace = self.memory_core.load_latest_profile_record(
            kind="relationship_trace",
            culture_id=str(world_context.get("culture_id") or "") or None,
            community_id=str(world_context.get("community_id") or "") or None,
            social_role=str(world_context.get("social_role") or "") or None,
            memory_anchor=str(world_context.get("place_memory_anchor") or "") or None,
            related_person_id=counterpart_person_id,
        )
        community_profile_trace = self.memory_core.load_latest_profile_record(
            kind="community_profile_trace",
            culture_id=str(world_context.get("culture_id") or "") or None,
            community_id=str(world_context.get("community_id") or "") or None,
        )
        context_shift_trace = self.memory_core.load_latest_profile_record(
            kind="context_shift_trace",
            culture_id=str(world_context.get("culture_id") or "") or None,
            community_id=str(world_context.get("community_id") or "") or None,
        )
        registry_person_state = _person_registry_person_state(
            registry_snapshot,
            counterpart_person_id,
        )
        working_memory_trace = self.memory_core.load_latest_profile_record(
            kind="working_memory_trace",
            culture_id=str(world_context.get("culture_id") or "") or None,
            community_id=str(world_context.get("community_id") or "") or None,
            social_role=str(world_context.get("social_role") or "") or None,
        )
        relation_seed_summary = _text_or_none(working_memory_seed.get("relation_seed_summary")) or _text_or_none((current_state or {}).get("relation_seed_summary"))
        partner_address_hint = _text_or_none(working_memory_seed.get("partner_address_hint")) or _text_or_none((current_state or {}).get("partner_address_hint"))
        partner_timing_hint = _text_or_none(working_memory_seed.get("partner_timing_hint")) or _text_or_none((current_state or {}).get("partner_timing_hint"))
        partner_stance_hint = _text_or_none(working_memory_seed.get("partner_stance_hint")) or _text_or_none((current_state or {}).get("partner_stance_hint"))
        partner_social_interpretation = _text_or_none(working_memory_seed.get("partner_social_interpretation")) or _text_or_none((current_state or {}).get("partner_social_interpretation"))
        merged_state = dict(identity_trace or {})
        merged_state.update(dict(relationship_trace or {}))
        merged_state.update(dict(community_profile_trace or {}))
        merged_state.update(dict(context_shift_trace or {}))
        merged_state.update(dict(registry_person_state or {}))
        merged_state.update(dict(working_memory_trace or {}))
        merged_state.update(dict(current_state or {}))
        merged_state.update(dict(working_memory_seed or {}))
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
        field_estimate = self.field_estimator_core.snapshot(
            current_state=current_state,
            observed_roughness=terrain.transition_roughness,
            observed_defensive_salience=peripersonal.defensive_salience,
        )
        working_memory = self.working_memory_core.snapshot(
            user_input=user_input,
            sensor_input=sensor_input,
            current_state=merged_state,
            relational_world=world_context,
            previous_trace=working_memory_trace,
        )
        if field_estimate.roughness_level >= 0.38 and talk_mode == "talk":
            talk_mode = "ask"
        if working_memory.pending_meaning >= 0.34 and talk_mode == "talk":
            talk_mode = "ask"
            route = "conscious"
        if working_memory.current_focus == "body" and talk_mode == "talk":
            talk_mode = "soothe"
        if working_memory.current_focus == "meaning" and temporal_pressure < 0.72:
            route = "conscious"
        if (
            working_memory.long_term_theme_strength >= 0.42
            and working_memory.current_focus in {"meaning", "place"}
            and talk_mode == "watch"
            and temporal_pressure < 0.62
        ):
            talk_mode = "ask"
        if (
            working_memory.long_term_theme_strength >= 0.5
            and working_memory.current_focus in {"meaning", "place"}
            and route == "reflex"
            and temporal_pressure < 0.8
        ):
            route = "conscious"
        prior_identity_update_strength = _float_from(current_state, "identity_update_strength", default=1.0)
        prior_social_update_strength = _float_from(current_state, "social_update_strength", default=1.0)
        prior_interaction_afterglow = _float_from(current_state, "interaction_afterglow", default=0.0)
        prior_afterglow_intent = _text_or_none((current_state or {}).get("interaction_afterglow_intent"))
        predicted_relation_bias = _clamp01(
            relationship.attachment * 0.34
            + relationship.familiarity * 0.22
            + relationship.trust_memory * 0.22
            + persistence.continuity_score * 0.12
            + persistence.social_grounding * 0.1
        )
        predicted_situation_state = SituationState(
            scene_mode="co_present" if counterpart_person_id else "ambient",
            repair_window_open=bool(working_memory.pending_meaning >= 0.32 or prior_afterglow_intent == "check_in"),
            shared_attention=_clamp01(predicted_relation_bias * 0.72 + relationship.attachment * 0.12),
            social_pressure=_clamp01(personality.caution_bias * 0.62 + persistence.recent_strain * 0.28),
            continuity_weight=predicted_relation_bias,
            current_phase="check_in" if counterpart_person_id and predicted_relation_bias >= 0.28 else "ongoing",
        )
        predicted_relational_mood = derive_relational_mood(
            affective_summary={
                "trust": development.trust_bias,
                "curiosity": personality.affiliation_bias,
                "arousal": _clamp01(0.34 + relationship.attachment * 0.12 + relationship.familiarity * 0.1),
                "social_tension": personality.caution_bias,
            },
            situation_state=predicted_situation_state,
            partner_address_hint=partner_address_hint,
            partner_timing_hint=partner_timing_hint,
            partner_stance_hint=partner_stance_hint,
        )
        predicted_utterance_stance = resolve_partner_utterance_stance(
            relation_bias_strength=predicted_relation_bias,
            related_person_ids=[counterpart_person_id] if counterpart_person_id else [],
            partner_address_hint=partner_address_hint,
            partner_timing_hint=partner_timing_hint,
            partner_stance_hint=partner_stance_hint,
        )
        predicted_nonverbal = compose_nonverbal_profile(
            utterance_stance=predicted_utterance_stance,
            affective_summary={
                "social_tension": personality.caution_bias,
                "arousal": _clamp01(0.34 + relationship.attachment * 0.12 + relationship.familiarity * 0.1),
            },
            situation_state=predicted_situation_state,
            relational_mood=predicted_relational_mood,
            partner_address_hint=partner_address_hint,
            partner_timing_hint=partner_timing_hint,
            partner_stance_hint=partner_stance_hint,
        )
        future_signal = _float_from(current_state, "future_signal", default=0.0)
        raw_core_axes = _core_state_axes(
            stress=stress,
            recovery_need=recovery_need,
            temporal_pressure=temporal_pressure,
            continuity_score=persistence.continuity_score,
            social_grounding=persistence.social_grounding,
            recent_strain=persistence.recent_strain,
            trust_bias=development.trust_bias,
            terrain_transition_roughness=field_estimate.roughness_level,
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
            terrain_transition_roughness=field_estimate.roughness_level,
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
            defensive_salience=field_estimate.defensive_level,
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
            terrain_transition_roughness=field_estimate.roughness_level,
            roughness_level=field_estimate.roughness_level,
            roughness_velocity=field_estimate.roughness_velocity,
            roughness_momentum=field_estimate.roughness_momentum,
            roughness_dwell=field_estimate.roughness_dwell,
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
            near_body_risk=peripersonal.near_body_risk,
            defensive_salience=peripersonal.defensive_salience,
            defensive_level=field_estimate.defensive_level,
            defensive_velocity=field_estimate.defensive_velocity,
            defensive_momentum=field_estimate.defensive_momentum,
            defensive_dwell=field_estimate.defensive_dwell,
            approach_confidence=peripersonal.approach_confidence,
            reachability=peripersonal.reachability,
            working_memory_pressure=working_memory.memory_pressure,
            unresolved_count=working_memory.unresolved_count,
            current_focus=working_memory.current_focus,
            pending_meaning=working_memory.pending_meaning,
            related_person_id=counterpart_person_id or "",
            related_person_ids=related_person_ids,
            relation_seed_summary=relation_seed_summary,
            partner_address_hint=partner_address_hint,
            partner_timing_hint=partner_timing_hint,
            partner_stance_hint=partner_stance_hint,
            partner_social_interpretation=partner_social_interpretation,
            long_term_theme_focus=working_memory.long_term_theme_focus,
            long_term_theme_anchor=working_memory.long_term_theme_anchor,
            long_term_theme_kind=working_memory.long_term_theme_kind,
            long_term_theme_summary=working_memory.long_term_theme_summary,
            long_term_theme_strength=working_memory.long_term_theme_strength,
            conscious_residue_focus=working_memory.conscious_residue_focus,
            conscious_residue_anchor=working_memory.conscious_residue_anchor,
            conscious_residue_summary=working_memory.conscious_residue_summary,
            conscious_residue_strength=working_memory.conscious_residue_strength,
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
                "counterpart_person_id": counterpart_person_id,
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
                "field_estimate": field_estimate.to_dict(),
                "working_memory": working_memory.to_dict(),
                "working_memory_trace": working_memory_trace,
                "working_memory_seed": dict(working_memory_seed),
                "predicted_relational_mood": {
                    "future_pull": predicted_relational_mood.future_pull,
                    "reverence": predicted_relational_mood.reverence,
                    "innocence": predicted_relational_mood.innocence,
                    "care": predicted_relational_mood.care,
                    "shared_world_pull": predicted_relational_mood.shared_world_pull,
                    "confidence_signal": predicted_relational_mood.confidence_signal,
                },
                "predicted_nonverbal": {
                    "gaze_mode": predicted_nonverbal.gaze_mode,
                    "pause_mode": predicted_nonverbal.pause_mode,
                    "proximity_mode": predicted_nonverbal.proximity_mode,
                    "silence_mode": predicted_nonverbal.silence_mode,
                    "gesture_mode": predicted_nonverbal.gesture_mode,
                    "cues": list(predicted_nonverbal.cues),
                },
                "predicted_utterance_stance": predicted_utterance_stance,
                "predicted_shared_attention": round(predicted_situation_state.shared_attention, 4),
                "predicted_distance_expectation": predicted_nonverbal.proximity_mode,
                "predicted_hesitation_tone": predicted_nonverbal.pause_mode,
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
        counterpart_person_id = _text_or_none(
            world_snapshot.get("person_id")
            or (current_state or {}).get("related_person_id")
        )

        identity_trace = self.memory_core.load_latest_identity_trace(
            culture_id=str(world_snapshot.get("culture_id") or "") or None,
            community_id=str(world_snapshot.get("community_id") or "") or None,
            social_role=str(world_snapshot.get("social_role") or "") or None,
            related_person_id=counterpart_person_id,
        )
        relationship_trace = self.memory_core.load_latest_profile_record(
            kind="relationship_trace",
            culture_id=str(world_snapshot.get("culture_id") or "") or None,
            community_id=str(world_snapshot.get("community_id") or "") or None,
            social_role=str(world_snapshot.get("social_role") or "") or None,
            memory_anchor=str(world_snapshot.get("place_memory_anchor") or "") or None,
            related_person_id=counterpart_person_id,
        )
        community_profile_trace = self.memory_core.load_latest_profile_record(
            kind="community_profile_trace",
            culture_id=str(world_snapshot.get("culture_id") or "") or None,
            community_id=str(world_snapshot.get("community_id") or "") or None,
        )
        context_shift_trace = self.memory_core.load_latest_profile_record(
            kind="context_shift_trace",
            culture_id=str(world_snapshot.get("culture_id") or "") or None,
            community_id=str(world_snapshot.get("community_id") or "") or None,
        )
        working_memory_trace = self.memory_core.load_latest_profile_record(
            kind="working_memory_trace",
            culture_id=str(world_snapshot.get("culture_id") or "") or None,
            community_id=str(world_snapshot.get("community_id") or "") or None,
            social_role=str(world_snapshot.get("social_role") or "") or None,
        )
        merged_state = dict(identity_trace or {})
        merged_state.update(dict(relationship_trace or {}))
        merged_state.update(dict(community_profile_trace or {}))
        merged_state.update(dict(context_shift_trace or {}))
        merged_state.update(dict(working_memory_trace or {}))
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
        initial_peripersonal = self.peripersonal_core.snapshot(
            relational_world=world_snapshot,
            current_state=merged_state,
            object_relation=object_relation.to_dict(),
        )
        field_estimate = self.field_estimator_core.snapshot(
            current_state=current_state,
            observed_roughness=terrain.transition_roughness,
            observed_defensive_salience=initial_peripersonal.defensive_salience,
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
            terrain_transition_roughness=field_estimate.roughness_level,
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
        merged_state["terrain_transition_roughness"] = field_estimate.roughness_level
        merged_state["roughness_level"] = field_estimate.roughness_level
        merged_state["roughness_velocity"] = field_estimate.roughness_velocity
        merged_state["roughness_momentum"] = field_estimate.roughness_momentum
        merged_state["roughness_dwell"] = field_estimate.roughness_dwell
        forgetting_snapshot = self.forgetting_core.snapshot(
            stress=_float_from(merged_state, "stress", default=0.0),
            recovery_need=_float_from(merged_state, "recovery_need", default=0.0),
            terrain_transition_roughness=field_estimate.roughness_level,
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
        field_estimate = self.field_estimator_core.snapshot(
            current_state=current_state,
            observed_roughness=terrain.transition_roughness,
            observed_defensive_salience=peripersonal.defensive_salience,
        )
        working_memory = self.working_memory_core.snapshot(
            user_input={"text": cue_text},
            sensor_input=current_state,
            current_state=merged_state,
            relational_world=world_snapshot,
            previous_trace=working_memory_trace,
        )
        merged_state["defensive_salience"] = field_estimate.defensive_level
        merged_state["defensive_level"] = field_estimate.defensive_level
        merged_state["defensive_velocity"] = field_estimate.defensive_velocity
        merged_state["defensive_momentum"] = field_estimate.defensive_momentum
        merged_state["defensive_dwell"] = field_estimate.defensive_dwell
        replay_signature = retrieval.get("working_memory_replay_signature")
        if not isinstance(replay_signature, Mapping):
            replay_signature = {}
        replay_signature_focus = _text_or_none(replay_signature.get("focus")) or _text_or_none((current_state or {}).get("working_memory_replay_focus"))
        replay_signature_anchor = _text_or_none(replay_signature.get("anchor")) or _text_or_none((current_state or {}).get("working_memory_replay_anchor"))
        replay_signature_strength = (
            _float_from(replay_signature, "strength", default=0.0)
            if isinstance(replay_signature, Mapping)
            else 0.0
        )
        if replay_signature_strength <= 0.0:
            replay_signature_strength = _float_from(current_state, "working_memory_replay_strength", default=0.0)
        semantic_seed_focus = _text_or_none((current_state or {}).get("semantic_seed_focus"))
        semantic_seed_anchor = _text_or_none((current_state or {}).get("semantic_seed_anchor"))
        semantic_seed_strength = _float_from(current_state, "semantic_seed_strength", default=0.0)
        relation_seed_summary = _text_or_none((current_state or {}).get("relation_seed_summary"))
        long_term_theme_focus = _text_or_none((current_state or {}).get("long_term_theme_focus"))
        long_term_theme_anchor = _text_or_none((current_state or {}).get("long_term_theme_anchor"))
        long_term_theme_kind = _text_or_none((current_state or {}).get("long_term_theme_kind"))
        long_term_theme_summary = _text_or_none((current_state or {}).get("long_term_theme_summary"))
        long_term_theme_strength = _float_from(current_state, "long_term_theme_strength", default=0.0)
        related_person_id = counterpart_person_id
        local_payload = self.memory_core.build_recall_payload(
            cue_text,
            limit=3,
            bias_context={
                "culture_id": world_snapshot.get("culture_id"),
                "community_id": world_snapshot.get("community_id"),
                "social_role": world_snapshot.get("social_role"),
                "related_person_id": related_person_id,
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
                "terrain_transition_roughness": field_estimate.roughness_level,
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
                "defensive_salience": field_estimate.defensive_level,
                "approach_confidence": peripersonal.approach_confidence,
                "roughness_level": field_estimate.roughness_level,
                "roughness_velocity": field_estimate.roughness_velocity,
                "roughness_momentum": field_estimate.roughness_momentum,
                "roughness_dwell": field_estimate.roughness_dwell,
                "defensive_level": field_estimate.defensive_level,
                "defensive_velocity": field_estimate.defensive_velocity,
                "defensive_momentum": field_estimate.defensive_momentum,
                "defensive_dwell": field_estimate.defensive_dwell,
                "reuse_trajectory": memory_orchestration["reuse_trajectory"],
                "interference_pressure": memory_orchestration["interference_pressure"],
                "consolidation_priority": memory_orchestration["consolidation_priority"],
                "prospective_memory_pull": memory_orchestration["prospective_memory_pull"],
                "kind_biases": development_biases,
                "working_memory_pressure": working_memory.memory_pressure,
                "current_focus": working_memory.current_focus,
                "pending_meaning": working_memory.pending_meaning,
                "replay_signature_focus": replay_signature_focus,
                "replay_signature_anchor": replay_signature_anchor,
                "replay_signature_strength": replay_signature_strength,
                "semantic_seed_focus": semantic_seed_focus,
                "semantic_seed_anchor": semantic_seed_anchor,
                "semantic_seed_strength": semantic_seed_strength,
                "relation_seed_summary": relation_seed_summary,
                "partner_social_interpretation": _text_or_none((current_state or {}).get("partner_social_interpretation")),
                "long_term_theme_focus": long_term_theme_focus,
                "long_term_theme_anchor": long_term_theme_anchor,
                "long_term_theme_summary": long_term_theme_summary,
                "long_term_theme_strength": long_term_theme_strength,
            },
        )
        if local_payload.get("record_id"):
            touched_record = self.memory_core.touch_record_usage(str(local_payload.get("record_id") or ""))
            if touched_record:
                local_payload["access_count"] = touched_record.get("access_count")
                local_payload["primed_weight"] = touched_record.get("primed_weight")
                local_payload["last_accessed_at"] = touched_record.get("last_accessed_at")
        if related_person_id:
            local_payload["counterpart_person_id"] = related_person_id
        if local_payload.get("hits"):
            retrieval.setdefault("inner_os_memory", local_payload.get("hits"))

        anchor = str(local_payload.get("memory_anchor") or cue_text[:160]).strip()
        recall_payload = {
            "memory_anchor": anchor,
            "cue_text": cue_text,
            "culture_id": world_snapshot.get("culture_id"),
            "community_id": world_snapshot.get("community_id"),
            "social_role": world_snapshot.get("social_role"),
            "working_memory_pressure": round(working_memory.memory_pressure, 4),
            "current_focus": working_memory.current_focus,
            "pending_meaning": round(working_memory.pending_meaning, 4),
            "unresolved_count": int(working_memory.unresolved_count),
            "replay_signature_focus": replay_signature_focus,
            "replay_signature_anchor": replay_signature_anchor,
            "replay_signature_strength": round(replay_signature_strength, 4),
            "semantic_seed_focus": semantic_seed_focus,
            "semantic_seed_anchor": semantic_seed_anchor,
            "semantic_seed_strength": round(semantic_seed_strength, 4),
            "relation_seed_summary": relation_seed_summary,
            "partner_social_interpretation": _text_or_none((current_state or {}).get("partner_social_interpretation")),
            "long_term_theme_focus": long_term_theme_focus,
            "long_term_theme_anchor": long_term_theme_anchor,
            "long_term_theme_kind": long_term_theme_kind,
            "long_term_theme_summary": long_term_theme_summary,
            "long_term_theme_strength": round(long_term_theme_strength, 4),
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
                    "related_person_id": local_payload.get("related_person_id"),
                    "counterpart_person_id": local_payload.get("counterpart_person_id"),
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
                    "defensive_salience": field_estimate.defensive_level,
                    "approach_confidence": peripersonal.approach_confidence,
                    "roughness_level": round(field_estimate.roughness_level, 4),
                    "roughness_velocity": round(field_estimate.roughness_velocity, 4),
                    "roughness_momentum": round(field_estimate.roughness_momentum, 4),
                    "roughness_dwell": round(field_estimate.roughness_dwell, 4),
                    "defensive_level": round(field_estimate.defensive_level, 4),
                    "defensive_velocity": round(field_estimate.defensive_velocity, 4),
                    "defensive_momentum": round(field_estimate.defensive_momentum, 4),
                    "defensive_dwell": round(field_estimate.defensive_dwell, 4),
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
                "community_profile_pressure": round(reinterpretation.community_profile_pressure, 4),
                "terrain_observed_roughness": round(terrain.transition_roughness, 4),
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

        memory_evidence_bundle = build_temporal_memory_evidence_bundle(
            cue_text=cue_text,
            current_state=current_state,
            world_snapshot=world_snapshot,
            recall_payload=recall_payload,
            retrieval_summary=retrieval,
        )
        qualia_membrane_temporal = derive_qualia_membrane_temporal_bias(
            memory_evidence_bundle=memory_evidence_bundle,
            current_state=current_state,
            world_snapshot=world_snapshot,
        )
        recall_payload.update(
            {
                "temporal_timeline_coherence": qualia_membrane_temporal.timeline_coherence,
                "temporal_reentry_pull": qualia_membrane_temporal.reentry_pull,
                "temporal_supersession_pressure": qualia_membrane_temporal.supersession_pressure,
                "temporal_continuity_pressure": qualia_membrane_temporal.continuity_pressure,
                "temporal_relation_reentry_pull": qualia_membrane_temporal.relation_reentry_pull,
                "temporal_membrane_mode": qualia_membrane_temporal.dominant_mode,
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
                "field_estimate": field_estimate.to_dict(),
                "identity_trace": identity_trace,
                "relationship_trace": relationship_trace,
                "community_profile_trace": community_profile_trace,
                "context_shift_trace": context_shift_trace,
                "working_memory_trace": working_memory_trace,
                "environment_pressure": environment_pressure.to_dict(),
                "transition_signal": transition_signal,
                "context_shift": transition_signal,
                "community_profile_trace": community_profile_trace,
                "context_shift_trace": context_shift_trace,
                "working_memory": working_memory.to_dict(),
                "relation_seed_summary": relation_seed_summary,
                "qualia_membrane_temporal": qualia_membrane_temporal.to_dict(),
            },
            memory_evidence_bundle=memory_evidence_bundle,
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
        attachment = _float_from(current_state, "attachment", default=0.42)
        familiarity = _float_from(current_state, "familiarity", default=0.35)
        trust_memory = _float_from(current_state, "trust_memory", default=0.45)
        continuity_score = _float_from(current_state, "continuity_score", default=0.48)
        social_grounding = _float_from(current_state, "social_grounding", default=0.44)
        recent_strain = _float_from(current_state, "recent_strain", default=0.32)
        culture_resonance = _float_from(current_state, "culture_resonance", default=0.0)
        community_resonance = _float_from(current_state, "community_resonance", default=0.0)
        terrain_transition_roughness = _float_from(current_state, "terrain_transition_roughness", default=0.0)
        roughness_level = _float_from(current_state, "roughness_level", default=terrain_transition_roughness)
        roughness_velocity = _float_from(current_state, "roughness_velocity", default=0.0)
        roughness_momentum = _float_from(current_state, "roughness_momentum", default=0.0)
        roughness_dwell = _float_from(current_state, "roughness_dwell", default=0.0)
        recalled_tentative_bias = _float_from(current_state, "recalled_tentative_bias", default=0.0)
        recalled_reinterpretation_mode = _text_or_none(current_state.get("recalled_reinterpretation_mode"))
        social_update_strength = _float_from(current_state, "social_update_strength", default=1.0)
        identity_update_strength = _float_from(current_state, "identity_update_strength", default=1.0)
        interaction_afterglow = _float_from(current_state, "interaction_afterglow", default=0.0)
        interaction_afterglow_intent = _text_or_none((current_state or {}).get("interaction_afterglow_intent"))
        replay_intensity = _float_from(current_state, "replay_intensity", default=0.0)
        anticipation_tension = _float_from(current_state, "anticipation_tension", default=0.0)
        stabilization_drive = _float_from(current_state, "stabilization_drive", default=0.0)
        relational_clarity = _float_from(current_state, "relational_clarity", default=0.0)
        meaning_inertia = _float_from(current_state, "meaning_inertia", default=0.0)
        working_memory_pressure = _float_from(current_state, "working_memory_pressure", default=0.0)
        current_focus = _text_or_none(current_state.get("current_focus")) or "ambient"
        pending_meaning = _float_from(current_state, "pending_meaning", default=0.0)
        long_term_theme_focus = _text_or_none(current_state.get("long_term_theme_focus"))
        long_term_theme_anchor = _text_or_none(current_state.get("long_term_theme_anchor"))
        long_term_theme_strength = _float_from(current_state, "long_term_theme_strength", default=0.0)
        long_term_theme_kind = _text_or_none(current_state.get("long_term_theme_kind")) or ""
        object_affordance_bias = _float_from(current_state, "object_affordance_bias", default=0.0)
        fragility_guard = _float_from(current_state, "fragility_guard", default=0.0)
        object_attachment = _float_from(current_state, "object_attachment", default=0.0)
        object_avoidance = _float_from(current_state, "object_avoidance", default=0.0)
        ritually_sensitive_bias = _float_from(current_state, "ritually_sensitive_bias", default=0.0)
        reachability = _float_from(current_state, "reachability", default=0.0)
        near_body_risk = _float_from(current_state, "near_body_risk", default=0.0)
        defensive_salience = _float_from(current_state, "defensive_salience", default=0.0)
        defensive_level = _float_from(current_state, "defensive_level", default=defensive_salience)
        defensive_velocity = _float_from(current_state, "defensive_velocity", default=0.0)
        defensive_momentum = _float_from(current_state, "defensive_momentum", default=0.0)
        defensive_dwell = _float_from(current_state, "defensive_dwell", default=0.0)
        approach_confidence = _float_from(current_state, "approach_confidence", default=0.0)
        related_person_id = _text_or_none(current_state.get("related_person_id"))
        related_person_ids = collect_related_person_ids(
            related_person_id,
            current_state.get("related_person_ids"),
            registry_snapshot=current_state.get("person_registry_snapshot"),
            limit=4,
        )
        if not related_person_id and related_person_ids:
            related_person_id = related_person_ids[0]
        partner_address_hint = _text_or_none(current_state.get("partner_address_hint")) or ""
        partner_timing_hint = _text_or_none(current_state.get("partner_timing_hint")) or ""
        partner_stance_hint = _text_or_none(current_state.get("partner_stance_hint")) or ""
        partner_social_interpretation = _text_or_none(current_state.get("partner_social_interpretation")) or ""
        person_specific_relief = 0.0
        if related_person_id:
            person_specific_relief = _clamp01(
                attachment * 0.12 + familiarity * 0.1 + trust_memory * 0.08
            )
        partner_style_relief = 0.0
        partner_style_caution = 0.0
        if partner_stance_hint == "familiar":
            partner_style_relief += 0.04
        elif partner_stance_hint == "respectful":
            partner_style_caution += 0.02
        if partner_timing_hint == "open":
            partner_style_relief += 0.03
        elif partner_timing_hint == "delayed":
            partner_style_caution += 0.05
        if partner_address_hint == "companion":
            partner_style_relief += 0.03
        elif partner_address_hint == "respectful":
            partner_style_caution += 0.02
        if partner_social_interpretation:
            partner_style_relief += 0.02 if "familiar" in partner_social_interpretation else 0.0
            partner_style_caution += 0.03 if "delayed" in partner_social_interpretation else 0.0
        relation_bias_strength = _clamp01(
            attachment * 0.34
            + familiarity * 0.22
            + trust_memory * 0.22
            + continuity_score * 0.12
            + social_grounding * 0.1
        )
        partner_utterance_stance = resolve_partner_utterance_stance(
            relation_bias_strength=relation_bias_strength,
            related_person_ids=[related_person_id] if related_person_id else [],
            partner_address_hint=partner_address_hint,
            partner_timing_hint=partner_timing_hint,
            partner_stance_hint=partner_stance_hint,
        )
        situation_state = SituationState(
            scene_mode="co_present" if related_person_id else "ambient",
            repair_window_open=bool(pending_meaning >= 0.32 or interaction_afterglow_intent == "check_in"),
            shared_attention=_clamp01(relation_bias_strength * 0.72 + person_specific_relief * 0.45),
            social_pressure=_clamp01(caution_bias * 0.6 + recent_strain * 0.4),
            continuity_weight=relation_bias_strength,
            current_phase="check_in" if related_person_id and relation_bias_strength >= 0.28 else "ongoing",
        )
        relational_mood = derive_relational_mood(
            affective_summary={
                "trust": trust_bias,
                "curiosity": affiliation_bias,
                "arousal": _clamp01(0.4 + person_specific_relief - partner_style_caution * 0.5),
                "social_tension": caution_bias,
            },
            situation_state=situation_state,
            partner_address_hint=partner_address_hint,
            partner_timing_hint=partner_timing_hint,
            partner_stance_hint=partner_stance_hint,
        )
        nonverbal_profile = compose_nonverbal_profile(
            utterance_stance=partner_utterance_stance,
            affective_summary={
                "social_tension": caution_bias,
                "arousal": _clamp01(0.4 + person_specific_relief - partner_style_caution * 0.5),
            },
            situation_state=situation_state,
            relational_mood=relational_mood,
            partner_address_hint=partner_address_hint,
            partner_timing_hint=partner_timing_hint,
            partner_stance_hint=partner_stance_hint,
        )
        observed_trace = summarize_interaction_trace(
            sensor_input=safety_signals,
            current_state=current_state,
        )
        live_regulation = derive_live_interaction_regulation(
            current_state=current_state,
            situation_state=situation_state,
            relational_mood=relational_mood,
            interaction_trace=observed_trace,
        )
        orchestration = orchestrate_interaction(
            current_risks=["danger"] if safety_bias > 0.32 else [],
            situation_state=situation_state,
            relational_mood=relational_mood,
            nonverbal_profile=nonverbal_profile,
            live_regulation=live_regulation,
        )
        stream_state = advance_interaction_stream(
            orchestration=orchestration,
            interaction_trace=observed_trace,
            previous=_mapping_or_empty(current_state.get("interaction_stream_state")),
        )
        surface_profile = derive_surface_profile(
            speech_act=str(access.intent or "report"),
            utterance_stance=partner_utterance_stance,
            orchestration=orchestration,
            live_regulation=live_regulation,
            nonverbal_profile=nonverbal_profile,
        )
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
            terrain_transition_roughness=roughness_level,
            recalled_tentative_bias=recalled_tentative_bias,
            anticipation_tension=anticipation_tension,
            stabilization_drive=stabilization_drive,
            relational_clarity=relational_clarity,
            meaning_inertia=meaning_inertia,
        )
        object_hold = _clamp01(fragility_guard * 0.22 + object_avoidance * 0.24 + ritually_sensitive_bias * 0.16 + defensive_level * 0.2 + near_body_risk * 0.12)
        object_release = _clamp01(object_affordance_bias * 0.08 + object_attachment * 0.06 + reachability * 0.08 + approach_confidence * 0.06)
        hesitation_bias = _clamp01(
            float(response_allocation["hold_back"])
            + object_hold
            - object_release
            + working_memory_pressure * 0.1
            + pending_meaning * 0.12
            - person_specific_relief
            - partner_style_relief
            + partner_style_caution
            - long_term_theme_strength * RESPONSE_GATE_WEIGHTS["hesitation_long_term_relief"]
        )
        allowed_surface_intensity = max(
            RESPONSE_GATE_WEIGHTS["surface_floor"],
            float(response_allocation["express_now"])
            - object_hold * 0.12
            + object_release * 0.08
            - pending_meaning * 0.05
            + person_specific_relief * 0.45
            + partner_style_relief * 0.35
            - partner_style_caution * 0.18
            + long_term_theme_strength * RESPONSE_GATE_WEIGHTS["surface_long_term_support"],
        )
        if recalled_reinterpretation_mode == "grounding_deferral":
            hesitation_bias = _clamp01(hesitation_bias + 0.08)
            allowed_surface_intensity = max(RESPONSE_GATE_WEIGHTS["surface_floor"], allowed_surface_intensity - 0.08)
        expression_hints = _expression_hints(
            terrain_transition_roughness=roughness_level,
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
            defensive_salience=defensive_level,
            reachability=reachability,
            long_term_theme_strength=long_term_theme_strength,
            long_term_theme_kind=long_term_theme_kind,
        )
        expression_hints["recalled_tentative_bias"] = round(recalled_tentative_bias, 4)
        expression_hints["recalled_reinterpretation_mode"] = recalled_reinterpretation_mode
        expression_hints["tentative_bias"] = round(_clamp01(max(float(expression_hints.get("tentative_bias", 0.0) or 0.0), recalled_tentative_bias * 0.9)), 4)
        expression_hints["express_now"] = round(float(response_allocation["express_now"]), 4)
        expression_hints["hold_back"] = round(float(response_allocation["hold_back"]), 4)
        if recalled_tentative_bias >= INTENT_THRESHOLDS["tentative_definitive_cutoff"]:
            expression_hints["avoid_definitive_interpretation"] = True
        if pending_meaning >= 0.32:
            expression_hints["clarify_first"] = True
        if recalled_reinterpretation_mode == "grounding_deferral":
            expression_hints["avoid_definitive_interpretation"] = True
            expression_hints["favor_grounded_observation"] = True
        access_payload = access.to_dict()
        if identity_update_strength <= INTENT_THRESHOLDS["identity_clarify"] and access_payload.get("intent") in {"engage", "remember", "answer"}:
            access_payload["intent"] = "clarify"
        elif recalled_reinterpretation_mode == "grounding_deferral" and access_payload.get("intent") in {"engage", "remember", "answer", "listen"}:
            access_payload["intent"] = "clarify"
        elif social_update_strength <= INTENT_THRESHOLDS["social_check_in"] and access_payload.get("intent") == "listen":
            access_payload["intent"] = "check_in"
        elif interaction_afterglow >= INTENT_THRESHOLDS["afterglow_redirect"] and access_payload.get("intent") in {"engage", "listen", "answer"}:
            access_payload["intent"] = "check_in" if interaction_afterglow_intent == "check_in" else "clarify"
        elif anticipation_tension >= INTENT_THRESHOLDS["anticipation_clarify"] and access_payload.get("intent") in {"engage", "answer"}:
            access_payload["intent"] = "clarify"
        elif stabilization_drive >= INTENT_THRESHOLDS["stabilization_check_in"] and access_payload.get("intent") == "listen":
            access_payload["intent"] = "check_in"
        elif pending_meaning >= 0.32 and access_payload.get("intent") in {"engage", "answer", "remember"}:
            access_payload["intent"] = "clarify"
        elif current_focus == "body" and access_payload.get("intent") in {"engage", "answer"}:
            access_payload["intent"] = "soften"
        expression_hints["social_update_strength"] = round(social_update_strength, 4)
        expression_hints["identity_update_strength"] = round(identity_update_strength, 4)
        expression_hints["interaction_afterglow"] = round(interaction_afterglow, 4)
        expression_hints["interaction_afterglow_intent"] = interaction_afterglow_intent
        expression_hints["working_memory_pressure"] = round(working_memory_pressure, 4)
        expression_hints["current_focus"] = current_focus
        expression_hints["pending_meaning"] = round(pending_meaning, 4)
        expression_hints["related_person_id"] = related_person_id
        expression_hints["related_person_ids"] = list(related_person_ids)
        expression_hints["person_specific_relief"] = round(person_specific_relief, 4)
        expression_hints["partner_address_hint"] = partner_address_hint
        expression_hints["partner_timing_hint"] = partner_timing_hint
        expression_hints["partner_stance_hint"] = partner_stance_hint
        expression_hints["partner_social_interpretation"] = partner_social_interpretation
        expression_hints["partner_style_relief"] = round(partner_style_relief, 4)
        expression_hints["partner_style_caution"] = round(partner_style_caution, 4)
        expression_hints["relational_future_pull"] = relational_mood.future_pull
        expression_hints["relational_reverence"] = relational_mood.reverence
        expression_hints["relational_innocence"] = relational_mood.innocence
        expression_hints["relational_care"] = relational_mood.care
        expression_hints["shared_world_pull"] = relational_mood.shared_world_pull
        expression_hints["confidence_signal"] = relational_mood.confidence_signal
        expression_hints["partner_utterance_stance"] = partner_utterance_stance
        expression_hints["nonverbal_gaze_mode"] = nonverbal_profile.gaze_mode
        expression_hints["nonverbal_pause_mode"] = nonverbal_profile.pause_mode
        expression_hints["nonverbal_proximity_mode"] = nonverbal_profile.proximity_mode
        expression_hints["nonverbal_silence_mode"] = nonverbal_profile.silence_mode
        expression_hints["nonverbal_gesture_mode"] = nonverbal_profile.gesture_mode
        expression_hints["past_loop_pull"] = live_regulation.past_loop_pull
        expression_hints["future_loop_pull"] = live_regulation.future_loop_pull
        expression_hints["fantasy_loop_pull"] = live_regulation.fantasy_loop_pull
        expression_hints["shared_attention_active"] = live_regulation.shared_attention_active
        expression_hints["strained_pause"] = live_regulation.strained_pause
        expression_hints["repair_window_open"] = live_regulation.repair_window_open
        expression_hints["distance_expectation"] = live_regulation.distance_expectation
        expression_hints["interaction_orchestration_mode"] = orchestration["orchestration_mode"]
        expression_hints["interaction_dominant_driver"] = orchestration["dominant_driver"]
        expression_hints["interaction_coherence_score"] = orchestration["coherence_score"]
        expression_hints["interaction_contact_readiness"] = orchestration["contact_readiness"]
        expression_hints["human_presence_signal"] = orchestration["human_presence_signal"]
        expression_hints["observed_trace_gaze_mode"] = observed_trace.gaze_mode
        expression_hints["observed_trace_pause_mode"] = observed_trace.pause_mode
        expression_hints["observed_trace_proximity_mode"] = observed_trace.proximity_mode
        expression_hints["stream_shared_attention_level"] = stream_state.shared_attention_level
        expression_hints["stream_shared_attention_window_mean"] = round(
            (sum(stream_state.shared_attention_window) / len(stream_state.shared_attention_window))
            if stream_state.shared_attention_window
            else 0.0,
            4,
        )
        expression_hints["stream_strained_pause_level"] = stream_state.strained_pause_level
        expression_hints["stream_strained_pause_window_mean"] = round(
            (sum(stream_state.strained_pause_window) / len(stream_state.strained_pause_window))
            if stream_state.strained_pause_window
            else 0.0,
            4,
        )
        expression_hints["stream_repair_window_open"] = stream_state.repair_window_open
        expression_hints["stream_repair_window_hold"] = stream_state.repair_window_hold
        expression_hints["stream_contact_readiness"] = stream_state.contact_readiness
        expression_hints["stream_human_presence_signal"] = stream_state.human_presence_signal
        expression_hints["stream_update_count"] = stream_state.update_count
        expression_hints["opening_pace_windowed"] = _opening_pace_windowed(
            strained_pause_window_mean=expression_hints["stream_strained_pause_window_mean"],
            repair_window_hold=stream_state.repair_window_hold,
        )
        expression_hints["return_gaze_expectation"] = _return_gaze_expectation(
            shared_attention_window_mean=expression_hints["stream_shared_attention_window_mean"],
            repair_window_hold=stream_state.repair_window_hold,
            relational_reverence=relational_mood.reverence,
        )
        expression_hints["surface_opening_delay"] = surface_profile["opening_delay"]
        expression_hints["surface_response_length"] = surface_profile["response_length"]
        expression_hints["surface_sentence_temperature"] = surface_profile["sentence_temperature"]
        expression_hints["surface_pause_insertion"] = surface_profile["pause_insertion"]
        expression_hints["surface_certainty_style"] = surface_profile["certainty_style"]
        expression_hints["surface_banter_move"] = str(surface_profile.get("banter_move") or "")
        expression_hints["surface_lexical_variation_mode"] = str(
            surface_profile.get("lexical_variation_mode") or ""
        )
        expression_hints["surface_group_register"] = str(
            surface_profile.get("group_register") or ""
        )
        scene_norm_pressure = _clamp01(
            norm_pressure
            + (0.28 if partner_stance_hint == "respectful" else 0.0)
            + (0.18 if partner_timing_hint == "delayed" else 0.0)
        )
        scene_privacy_level = _clamp01(
            _float_from(
                current_state,
                "privacy_level",
                default=max(0.0, min(1.0, 0.48 + relation_bias_strength * 0.28 - scene_norm_pressure * 0.22)),
            )
        )
        scene_task_phase = (
            str(current_state.get("task_phase") or "")
            or (
                "repair"
                if (
                    live_regulation.repair_window_open
                    or observed_trace.repair_signal >= 0.42
                    or (recent_strain >= 0.3 and observed_trace.shared_attention <= 0.26)
                )
                else (
                    "coordination"
                    if (
                        orchestration["orchestration_mode"] == "advance"
                        and "future_open" in partner_social_interpretation
                        and partner_stance_hint != "respectful"
                        and partner_timing_hint != "delayed"
                    )
                    else "ongoing"
                )
            )
        )
        scene_state = derive_scene_state(
            place_mode=str(current_state.get("place_mode") or "unspecified"),
            privacy_level=scene_privacy_level,
            social_topology=str(current_state.get("social_topology") or ("one_to_one" if related_person_id else "ambient")),
            task_phase=scene_task_phase,
            temporal_phase=str(current_state.get("temporal_phase") or situation_state.current_phase or "ongoing"),
            norm_pressure=scene_norm_pressure,
            safety_margin=_clamp01(
                _float_from(current_state, "safety_margin", default=max(0.0, 0.82 - safety_bias * 0.46 - recent_strain * 0.18))
            ),
            environmental_load=_clamp01(
                _float_from(
                    current_state,
                    "environmental_load",
                    default=max(stress * 0.48, recovery_need * 0.4, recent_strain * 0.38, safety_bias * 0.44),
                )
            ),
            mobility_context=str(current_state.get("mobility_context") or "stationary"),
            current_risks=["danger"] if safety_bias > 0.32 else [],
            active_goals=(["repair"] if scene_task_phase == "repair" else (["coordinate"] if scene_task_phase == "coordination" else [])),
        )
        affect_blend_state = derive_affect_blend_state(
            affective_summary={
                "arousal": _float_from(current_state, "voice_level", default=0.0),
                "social_tension": recent_strain,
            },
            relational_mood=relational_mood,
            live_regulation=live_regulation,
            situation_state=situation_state,
            scene_state=scene_state,
            stress=stress,
            recovery_need=recovery_need,
            safety_bias=safety_bias,
            relation_bias_strength=relation_bias_strength,
            temporal_membrane_bias={
                "timeline_coherence": _float_from(current_state, "temporal_timeline_coherence", default=0.0),
                "reentry_pull": _float_from(current_state, "temporal_reentry_pull", default=0.0),
                "supersession_pressure": _float_from(current_state, "temporal_supersession_pressure", default=0.0),
                "continuity_pressure": _float_from(current_state, "temporal_continuity_pressure", default=0.0),
                "relation_reentry_pull": _float_from(current_state, "temporal_relation_reentry_pull", default=0.0),
                "dominant_mode": str(current_state.get("temporal_membrane_mode") or "").strip(),
            },
        )
        constraint_field = derive_constraint_field(
            scene_state=scene_state,
            affect_blend=affect_blend_state,
            stress=stress,
            recovery_need=recovery_need,
            safety_bias=safety_bias,
            recent_strain=recent_strain,
            current_risks=["danger"] if safety_bias > 0.32 else [],
        )
        qualia_kernel = self.runtime_qualia_adapter.step(
            current_state=current_state,
            safety_signals=safety_signals,
            prev_qualia=current_state.get("prev_qualia") or [],
            prev_habituation=current_state.get("prev_qualia_habituation") or [],
            prev_protection_grad_x=current_state.get("prev_protection_grad_x") or [],
            dt=_float_from(current_state, "kernel_dt", default=1.0),
        )
        qualia_state = qualia_kernel.qualia_state.to_dict()
        qualia_state["axis_labels"] = list(qualia_kernel.axis_labels)
        affective_position_dim = _default_affective_position_dim(qualia_kernel.axis_labels)
        prev_affective_position = _coerce_affective_position_state(
            current_state.get("prev_affective_position"),
            default_dim=affective_position_dim,
        )
        affective_terrain_state = _coerce_affective_terrain_state(
            current_state.get("affective_terrain_state"),
            position_dim=prev_affective_position.position_dim,
        )
        affective_position = BasicAffectiveLocalizer(
            position_dim=affective_terrain_state.position_dim
        ).localize(
            estimate=qualia_kernel.estimate,
            health=qualia_kernel.health,
            qualia_state=qualia_kernel.qualia_state,
            memory=_build_affective_memory_echo(current_state),
            prev_position=prev_affective_position,
            dt=_float_from(current_state, "kernel_dt", default=1.0),
        )
        terrain_readout = BasicAffectiveTerrain().read(
            affective_terrain_state,
            affective_position,
        )
        dot_seeds = derive_dot_seeds(
            qualia_state=qualia_state,
            current_state=current_state,
            current_text=_text_or_none(draft.get("text")) or "",
            current_focus=current_focus,
        )
        association_graph = BasicAssociationGraph().build(
            dot_seeds=dot_seeds,
            previous_state=coerce_association_graph_state(current_state.get("association_graph_state")),
            association_reweighting_bias=_float_from(current_state, "association_reweighting_bias", default=0.0),
            insight_reframing_bias=_float_from(current_state, "insight_reframing_bias", default=0.0),
            insight_class_focus=_text_or_none(current_state.get("insight_class_focus")) or "",
        )
        insight_event = BasicInsightDetector().detect(
            dot_seeds=dot_seeds,
            association_graph=association_graph,
            qualia_trust=float(qualia_kernel.qualia_state.trust_applied),
        )
        contact_field = derive_contact_field(
            affect_blend_state=affect_blend_state.to_dict(),
            constraint_field=constraint_field.to_dict(),
            scene_state=asdict(scene_state),
            current_focus=current_focus,
            reportable_facts=[current_focus] if current_focus and current_focus != "ambient" else [],
            current_risks=["danger"] if safety_bias > 0.32 else [],
            related_person_ids=[related_person_id] if related_person_id else [],
            memory_anchor=memory_anchor or current_focus,
            previous_residue=_float_from(current_state, "conscious_residue_strength", default=0.0),
        )
        contact_dynamics = advance_contact_dynamics(
            contact_field=contact_field.to_dict(),
            previous_dynamics=_mapping_or_none(current_state.get("contact_dynamics")),
            previous_workspace=_mapping_or_none(current_state.get("conscious_workspace")),
            previous_residue=_float_from(current_state, "conscious_residue_strength", default=0.0),
        )
        access_projection = project_access_regions(
            contact_field=contact_field.to_dict(),
            contact_dynamics=contact_dynamics.to_dict(),
            affect_blend_state=affect_blend_state.to_dict(),
            constraint_field=constraint_field.to_dict(),
            qualia_state=qualia_state,
            terrain_readout=terrain_readout.to_dict(),
            insight_event=insight_event.to_dict(),
        )
        contact_reflection_state = derive_contact_reflection_state(
            contact_field=contact_field.to_dict(),
            contact_dynamics=contact_dynamics.to_dict(),
            access_projection=access_projection.to_dict(),
            constraint_field=constraint_field.to_dict(),
            current_risks=["danger"] if safety_bias > 0.32 else [],
        )
        access_dynamics = advance_access_dynamics(
            access_projection=access_projection.to_dict(),
            previous_access_dynamics=_mapping_or_none(current_state.get("access_dynamics")),
            previous_workspace=_mapping_or_none(current_state.get("conscious_workspace")),
            previous_residue=_float_from(current_state, "conscious_residue_strength", default=0.0),
            current_risks=["danger"] if safety_bias > 0.32 else [],
        )
        interaction_option_candidates = generate_interaction_option_candidates(
            scene_state=scene_state,
            situation_state=situation_state,
            relational_mood=relational_mood,
            live_regulation=live_regulation,
            constraint_field=constraint_field.to_dict(),
        )
        provisional_workspace = ignite_conscious_workspace(
            affect_blend=affect_blend_state,
            constraint_field=constraint_field,
            current_focus=current_focus,
            reportable_facts=[current_focus] if current_focus and current_focus != "ambient" else [],
            current_risks=["danger"] if safety_bias > 0.32 else [],
            related_person_ids=[related_person_id] if related_person_id else [],
            interaction_option_candidates=interaction_option_candidates,
            memory_anchor=memory_anchor or current_focus,
            scene_state=asdict(scene_state),
            contact_field=contact_field.to_dict(),
            contact_dynamics=contact_dynamics.to_dict(),
            access_projection=access_projection.to_dict(),
            access_dynamics=access_dynamics.to_dict(),
            previous_workspace=_mapping_or_none(current_state.get("conscious_workspace")),
        )
        resonance_evaluation = evaluate_interaction_resonance(
            scene_state=scene_state,
            affect_blend=affect_blend_state,
            constraint_field=constraint_field,
            conscious_workspace=provisional_workspace,
            interaction_option_candidates=interaction_option_candidates,
            current_risks=["danger"] if safety_bias > 0.32 else [],
        )
        interaction_option_candidates = rerank_interaction_option_candidates(
            interaction_option_candidates=interaction_option_candidates,
            resonance_evaluation=resonance_evaluation,
        )
        conscious_workspace = ignite_conscious_workspace(
            affect_blend=affect_blend_state,
            constraint_field=constraint_field,
            current_focus=current_focus,
            reportable_facts=[current_focus] if current_focus and current_focus != "ambient" else [],
            current_risks=["danger"] if safety_bias > 0.32 else [],
            related_person_ids=[related_person_id] if related_person_id else [],
            interaction_option_candidates=interaction_option_candidates,
            memory_anchor=memory_anchor or current_focus,
            scene_state=asdict(scene_state),
            contact_field=contact_field.to_dict(),
            contact_dynamics=contact_dynamics.to_dict(),
            access_projection=access_projection.to_dict(),
            access_dynamics=access_dynamics.to_dict(),
            previous_workspace=provisional_workspace.to_dict(),
        )
        relation_context = {
            "relation_bias_strength": relation_bias_strength,
            "recent_strain": recent_strain,
            "trust_memory": _float_from(current_state, "trust_memory", default=0.0),
            "familiarity": _float_from(current_state, "familiarity", default=0.0),
            "attachment": _float_from(current_state, "attachment", default=0.0),
            "partner_timing_hint": partner_timing_hint,
            "partner_stance_hint": partner_stance_hint,
            "partner_social_interpretation": _text_or_none(current_state.get("partner_social_interpretation")) or "",
        }
        memory_context = {
            "memory_anchor": _text_or_none(current_state.get("memory_anchor")) or current_focus,
            "relation_seed_summary": _text_or_none(current_state.get("relation_seed_summary")) or "",
            "long_term_theme_summary": _text_or_none(current_state.get("long_term_theme_summary")) or "",
            "conscious_residue_summary": _text_or_none(current_state.get("conscious_residue_summary")) or "",
        }
        conversational_objects = derive_conversational_objects(
            current_text=_text_or_none(draft.get("text")) or "",
            current_focus=current_focus,
            reportable_facts=[current_focus] if current_focus and current_focus != "ambient" else [],
            scene_state=asdict(scene_state),
            relation_context=relation_context,
            memory_context=memory_context,
            affect_blend_state=affect_blend_state.to_dict(),
            constraint_field=constraint_field.to_dict(),
            conscious_workspace=conscious_workspace.to_dict(),
            resonance_evaluation=resonance_evaluation.to_dict(),
        )
        object_operations = derive_object_operations(
            conversational_objects=conversational_objects.to_dict(),
            scene_state=asdict(scene_state),
            relation_context=relation_context,
            memory_context=memory_context,
            resonance_evaluation=resonance_evaluation.to_dict(),
            constraint_field=constraint_field.to_dict(),
            conscious_workspace=conscious_workspace.to_dict(),
            interaction_option_candidates=[asdict(candidate) for candidate in interaction_option_candidates],
        )
        interaction_effects = derive_interaction_effects(
            conversational_objects=conversational_objects.to_dict(),
            object_operations=object_operations.to_dict(),
            resonance_evaluation=resonance_evaluation.to_dict(),
            constraint_field=constraint_field.to_dict(),
        )
        interaction_judgement_view = derive_interaction_judgement_view(
            current_text=_text_or_none(draft.get("text")) or "",
            reportable_facts=[current_focus] if current_focus and current_focus != "ambient" else [],
            conversational_objects=conversational_objects.to_dict(),
            object_operations=object_operations.to_dict(),
            interaction_effects=interaction_effects.to_dict(),
            resonance_evaluation=resonance_evaluation.to_dict(),
        )
        interaction_judgement_summary = derive_interaction_judgement_summary(
            interaction_judgement_view=interaction_judgement_view.to_dict(),
            conversational_objects=conversational_objects.to_dict(),
            object_operations=object_operations.to_dict(),
            interaction_effects=interaction_effects.to_dict(),
        )
        interaction_condition_report = build_interaction_condition_report(
            scene_state=asdict(scene_state),
            resonance_evaluation=resonance_evaluation.to_dict(),
            relation_context=relation_context,
            memory_context=memory_context,
        )
        reference_cases = select_same_utterance_audit_cases(
            current_state.get("interaction_audit_casebook"),
            current_text=_text_or_none(draft.get("text")) or "",
        )
        interaction_inspection_report = build_interaction_inspection_report(
            {
                "current_case": interaction_judgement_summary.to_dict(),
                **dict(reference_cases.get("judgement_summary_cases") or {}),
            }
        )
        interaction_audit_bundle = build_interaction_audit_bundle(
            interaction_judgement_summary=interaction_judgement_summary.to_dict(),
            interaction_condition_report=interaction_condition_report.to_dict(),
            interaction_inspection_report=interaction_inspection_report.to_dict(),
            conversational_objects=conversational_objects.to_dict(),
            object_operations=object_operations.to_dict(),
            interaction_effects=interaction_effects.to_dict(),
            resonance_evaluation=resonance_evaluation.to_dict(),
        )
        current_audit_case = build_interaction_audit_case_entry(
            observed_text=_text_or_none(draft.get("text")) or "",
            judgement_summary=interaction_judgement_summary.to_dict(),
            audit_bundle=interaction_audit_bundle.to_dict(),
            scene_state=asdict(scene_state),
            relation_context=relation_context,
            memory_context=memory_context,
        )
        interaction_audit_casebook = update_interaction_audit_casebook(
            current_state.get("interaction_audit_casebook"),
            current_audit_case,
        )
        interaction_audit_report = build_interaction_audit_report(
            {
                "current_case": interaction_audit_bundle.to_dict(),
                **dict(reference_cases.get("audit_bundle_cases") or {}),
            }
        )
        expression_hints["scene_state"] = asdict(scene_state)
        expression_hints["scene_family"] = scene_state.scene_family
        expression_hints["interaction_option_candidates"] = [asdict(candidate) for candidate in interaction_option_candidates]
        expression_hints["contact_field"] = contact_field.to_dict()
        expression_hints["contact_dynamics"] = contact_dynamics.to_dict()
        expression_hints["contact_dynamics_mode"] = contact_dynamics.dynamics_mode
        expression_hints["contact_reflection_state"] = contact_reflection_state.to_dict()
        expression_hints["qualia_state"] = qualia_state
        expression_hints["qualia_axis_labels"] = list(qualia_kernel.axis_labels)
        expression_hints["qualia_estimator_health"] = qualia_kernel.health.to_dict()
        expression_hints["qualia_protection_grad_x"] = qualia_kernel.protection_grad_x.astype("float32").tolist()
        qualia_planner_view = build_qualia_planner_view_hint(expression_hints)
        if qualia_planner_view is not None:
            expression_hints["qualia_planner_view"] = qualia_planner_view
        expression_hints = build_expression_hints_from_gate_result(
            expression_hints,
            existing_hints=expression_hints,
            expected_source="shared",
        )
        protection_mode = derive_protection_mode(
            terrain_readout=terrain_readout,
            affective_position=affective_position.to_dict(),
            self_state={
                "stress": _float_from(current_state, "stress", default=0.0),
                "recovery_need": _float_from(current_state, "recovery_need", default=0.0),
                "continuity_score": _float_from(current_state, "continuity_score", default=0.0),
                "recent_strain": recent_strain,
            },
            temperament_estimate=derive_temperament_estimate(current_state).to_dict(),
            workspace=conscious_workspace.to_dict(),
            qualia_planner_view=expression_hints.get("qualia_planner_view"),
            insight_reframing_bias=_float_from(current_state, "insight_reframing_bias", default=0.0),
            insight_class_focus=_text_or_none(current_state.get("insight_class_focus")) or "",
        )
        expression_hints["qualia_trust"] = float(qualia_kernel.qualia_state.trust_applied)
        expression_hints["qualia_degraded"] = bool(qualia_kernel.qualia_state.degraded)
        expression_hints["dot_seeds"] = dot_seeds.to_dict()
        expression_hints["association_graph"] = association_graph.to_dict()
        expression_hints["association_graph_winner_margin"] = float(association_graph.winner_margin)
        expression_hints["association_graph_dominant_inputs"] = list(association_graph.dominant_inputs)
        expression_hints["insight_event"] = insight_event.to_dict()
        expression_hints["affective_position"] = affective_position.to_dict()
        expression_hints["affective_position_confidence"] = float(affective_position.confidence)
        expression_hints["affective_terrain_state"] = affective_terrain_state.to_dict()
        expression_hints["terrain_readout"] = terrain_readout.to_dict()
        expression_hints["terrain_active_patch_label"] = terrain_readout.active_patch_label
        expression_hints["protection_mode"] = protection_mode.to_dict()
        expression_hints["protection_mode_name"] = protection_mode.mode
        expression_hints["protection_mode_strength"] = float(protection_mode.strength)
        expression_hints["access_projection"] = access_projection.to_dict()
        expression_hints["access_dynamics"] = access_dynamics.to_dict()
        expression_hints["access_dynamics_mode"] = access_dynamics.dynamics_mode
        expression_hints["affect_blend_state"] = affect_blend_state.to_dict()
        expression_hints["constraint_field"] = constraint_field.to_dict()
        expression_hints["conscious_workspace"] = conscious_workspace.to_dict()
        expression_hints["conscious_workspace_mode"] = conscious_workspace.workspace_mode
        expression_hints["conscious_workspace_reportable_slice"] = list(conscious_workspace.reportable_slice)
        expression_hints["conscious_workspace_withheld_slice"] = list(conscious_workspace.withheld_slice)
        expression_hints["conscious_workspace_actionable_slice"] = list(conscious_workspace.actionable_slice)
        expression_hints["conscious_workspace_ignition_phase"] = conscious_workspace.ignition_phase
        expression_hints["conscious_workspace_slot_scores"] = dict(conscious_workspace.slot_scores)
        expression_hints["conscious_workspace_winner_margin"] = float(conscious_workspace.winner_margin)
        expression_hints["conscious_workspace_dominant_inputs"] = list(conscious_workspace.dominant_inputs)
        expression_hints["conversational_objects"] = conversational_objects.to_dict()
        expression_hints["conversational_object_labels"] = list(conversational_objects.active_labels)
        expression_hints["conversational_object_pressure_balance"] = conversational_objects.pressure_balance
        expression_hints["object_operations"] = object_operations.to_dict()
        expression_hints["object_operation_question_budget"] = object_operations.question_budget
        expression_hints["object_operation_question_pressure"] = object_operations.question_pressure
        expression_hints["object_operation_defer_dominance"] = object_operations.defer_dominance
        expression_hints["interaction_effects"] = interaction_effects.to_dict()
        expression_hints["interaction_judgement_view"] = interaction_judgement_view.to_dict()
        expression_hints["interaction_judgement_summary"] = interaction_judgement_summary.to_dict()
        expression_hints["interaction_condition_report"] = interaction_condition_report.to_dict()
        expression_hints["interaction_inspection_report"] = interaction_inspection_report.to_dict()
        expression_hints["interaction_audit_bundle"] = interaction_audit_bundle.to_dict()
        expression_hints["interaction_audit_casebook"] = interaction_audit_casebook
        expression_hints["interaction_audit_report"] = interaction_audit_report.to_dict()
        expression_hints["interaction_audit_reference_case_ids"] = list(
            reference_cases.get("reference_case_ids") or []
        )
        expression_hints["interaction_audit_reference_case_meta"] = dict(
            reference_cases.get("reference_case_meta") or {}
        )
        expression_hints["resonance_evaluation"] = resonance_evaluation.to_dict()
        if interaction_option_candidates:
            expression_hints["top_interaction_option_family"] = interaction_option_candidates[0].family_id
        interaction_policy_packet = derive_interaction_policy_packet(
            dialogue_act=str(access_payload.get("intent") or "report"),
            current_focus=current_focus,
            current_risks=["danger"] if safety_bias > 0.32 else [],
            reportable_facts=[current_focus] if current_focus and current_focus != "ambient" else [],
            relation_bias_strength=relation_bias_strength,
            related_person_ids=related_person_ids,
            partner_address_hint=partner_address_hint,
            partner_timing_hint=partner_timing_hint,
            partner_stance_hint=partner_stance_hint,
            partner_social_interpretation=_text_or_none(current_state.get("partner_social_interpretation")) or "",
            recent_strain=recent_strain,
            orchestration=orchestration,
            surface_profile={
                **surface_profile,
                "opening_pace_windowed": expression_hints["opening_pace_windowed"],
                "return_gaze_expectation": expression_hints["return_gaze_expectation"],
            },
            live_regulation=live_regulation,
            scene_state=asdict(scene_state),
            interaction_option_candidates=[asdict(candidate) for candidate in interaction_option_candidates],
            affect_blend_state=affect_blend_state.to_dict(),
            constraint_field=constraint_field.to_dict(),
            conscious_workspace=conscious_workspace.to_dict(),
            resonance_evaluation=resonance_evaluation.to_dict(),
            conversational_objects=conversational_objects.to_dict(),
            object_operations=object_operations.to_dict(),
            interaction_effects=interaction_effects.to_dict(),
            interaction_judgement_view=interaction_judgement_view.to_dict(),
            qualia_planner_view=expression_hints.get("qualia_planner_view"),
            affective_position=expression_hints.get("affective_position"),
            terrain_readout=expression_hints.get("terrain_readout"),
            protection_mode=expression_hints.get("protection_mode"),
            insight_event=expression_hints.get("insight_event"),
            insight_reframing_bias=_float_from(current_state, "insight_reframing_bias", default=0.0),
            insight_class_focus=_text_or_none(current_state.get("insight_class_focus")) or "",
            association_reweighting_focus=_text_or_none(current_state.get("association_reweighting_focus")) or "",
            association_reweighting_reason=_text_or_none(current_state.get("association_reweighting_reason")) or "",
            insight_terrain_shape_target=_text_or_none(current_state.get("insight_terrain_shape_target")) or "",
            self_state=current_state,
        )
        recent_dialogue_state = derive_recent_dialogue_state(
            str(current_state.get("surface_user_text") or draft.get("text") or "").strip(),
            current_state.get("recent_dialogue_history"),
            interaction_policy=interaction_policy_packet,
        ).to_dict()
        interaction_policy_packet["recent_dialogue_state"] = recent_dialogue_state
        discussion_thread_state = derive_discussion_thread_state(
            str(current_state.get("surface_user_text") or draft.get("text") or "").strip(),
            current_state.get("recent_dialogue_history"),
            recent_dialogue_state=recent_dialogue_state,
            interaction_policy=interaction_policy_packet,
        ).to_dict()
        interaction_policy_packet["discussion_thread_state"] = discussion_thread_state
        issue_state = derive_issue_state(
            str(current_state.get("surface_user_text") or draft.get("text") or "").strip(),
            current_state.get("recent_dialogue_history"),
            discussion_thread_state=discussion_thread_state,
            recent_dialogue_state=recent_dialogue_state,
            interaction_policy=interaction_policy_packet,
        ).to_dict()
        interaction_policy_packet["issue_state"] = issue_state
        interaction_policy_packet["contact_reflection_state"] = contact_reflection_state.to_dict()
        green_kernel_composition = build_green_kernel_composition(
            temporal_membrane_bias=expression_hints.get("qualia_membrane_temporal"),
            memory_evidence_bundle=expression_hints.get("temporal_memory_evidence_bundle"),
            affect_blend_state=affect_blend_state.to_dict(),
            recent_dialogue_state=recent_dialogue_state,
            discussion_thread_state=discussion_thread_state,
            issue_state=issue_state,
            contact_reflection_state=contact_reflection_state.to_dict(),
            boundary_transform={},
            residual_reflection={},
            autobiographical_thread={
                "mode": str(interaction_policy_packet.get("autobiographical_thread_mode") or ""),
                "anchor": str(interaction_policy_packet.get("autobiographical_thread_anchor") or ""),
                "focus": str(interaction_policy_packet.get("autobiographical_thread_focus") or ""),
                "strength": interaction_policy_packet.get("autobiographical_thread_strength") or 0.0,
                "reasons": list(interaction_policy_packet.get("autobiographical_thread_reasons") or []),
            },
        ).to_dict()
        interaction_policy_packet["green_kernel_composition"] = green_kernel_composition
        conversation_contract = build_conversation_contract(
            conversational_objects=conversational_objects.to_dict(),
            object_operations=object_operations.to_dict(),
            interaction_effects=interaction_effects.to_dict(),
            interaction_judgement_summary=interaction_judgement_summary.to_dict(),
            interaction_condition_report=interaction_condition_report.to_dict(),
            interaction_policy=interaction_policy_packet,
        )
        interaction_policy_packet["conversation_contract"] = conversation_contract
        action_posture = derive_action_posture(interaction_policy_packet)
        actuation_plan = derive_actuation_plan(interaction_policy_packet, action_posture)
        expression_hints["conversation_contract"] = conversation_contract
        expression_hints["interaction_policy_packet"] = interaction_policy_packet
        expression_hints["recent_dialogue_state"] = recent_dialogue_state
        expression_hints["discussion_thread_state"] = discussion_thread_state
        expression_hints["issue_state"] = issue_state
        expression_hints["interaction_policy_contact_reflection_state"] = contact_reflection_state.to_dict()
        expression_hints["green_kernel_composition"] = green_kernel_composition
        expression_hints["interaction_policy_strategy"] = interaction_policy_packet["response_strategy"]
        expression_hints["interaction_policy_opening_move"] = interaction_policy_packet["opening_move"]
        expression_hints["interaction_policy_followup_move"] = interaction_policy_packet["followup_move"]
        expression_hints["interaction_policy_closing_move"] = interaction_policy_packet["closing_move"]
        expression_hints["interaction_policy_disclosure_depth"] = interaction_policy_packet["disclosure_depth"]
        expression_hints["interaction_policy_memory_write_priority"] = interaction_policy_packet["memory_write_priority"]
        expression_hints["interaction_policy_memory_write_class"] = interaction_policy_packet["memory_write_class"]
        expression_hints["interaction_policy_memory_write_class_reason"] = interaction_policy_packet["memory_write_class_reason"]
        expression_hints["interaction_policy_memory_write_class_bias"] = interaction_policy_packet["memory_write_class_bias"]
        expression_hints["interaction_policy_protection_mode_decision"] = interaction_policy_packet["protection_mode_decision"]
        expression_hints["interaction_policy_body_recovery_guard"] = interaction_policy_packet["body_recovery_guard"]
        expression_hints["interaction_policy_body_homeostasis_state"] = interaction_policy_packet["body_homeostasis_state"]
        expression_hints["interaction_policy_homeostasis_budget_state"] = interaction_policy_packet["homeostasis_budget_state"]
        expression_hints["interaction_policy_initiative_readiness"] = interaction_policy_packet["initiative_readiness"]
        expression_hints["interaction_policy_agenda_state"] = interaction_policy_packet["agenda_state"]
        expression_hints["interaction_policy_agenda_window_state"] = interaction_policy_packet["agenda_window_state"]
        expression_hints["interaction_policy_commitment_state"] = interaction_policy_packet["commitment_state"]
        expression_hints["interaction_policy_learning_mode_state"] = interaction_policy_packet["learning_mode_state"]
        expression_hints["interaction_policy_social_experiment_loop_state"] = interaction_policy_packet["social_experiment_loop_state"]
        expression_hints["interaction_policy_identity_arc_kind"] = interaction_policy_packet["identity_arc_kind"]
        expression_hints["interaction_policy_identity_arc_phase"] = interaction_policy_packet["identity_arc_phase"]
        expression_hints["interaction_policy_identity_arc_summary"] = interaction_policy_packet["identity_arc_summary"]
        expression_hints["interaction_policy_identity_arc_open_tension"] = interaction_policy_packet["identity_arc_open_tension"]
        expression_hints["interaction_policy_identity_arc_stability"] = interaction_policy_packet["identity_arc_stability"]
        expression_hints["interaction_policy_relational_style_memory_state"] = interaction_policy_packet["relational_style_memory_state"]
        expression_hints["interaction_policy_cultural_conversation_state"] = interaction_policy_packet["cultural_conversation_state"]
        expression_hints["interaction_policy_expressive_style_state"] = interaction_policy_packet["expressive_style_state"]
        expression_hints["interaction_policy_lightness_budget_state"] = interaction_policy_packet["lightness_budget_state"]
        expression_hints["interaction_policy_relational_continuity_state"] = interaction_policy_packet["relational_continuity_state"]
        expression_hints["interaction_policy_relation_competition_state"] = interaction_policy_packet["relation_competition_state"]
        expression_hints["interaction_policy_social_topology_state"] = interaction_policy_packet["social_topology_state"]
        expression_hints["interaction_policy_active_relation_table"] = interaction_policy_packet["active_relation_table"]
        expression_hints["interaction_policy_association_reweighting_focus"] = interaction_policy_packet["association_reweighting_focus"]
        expression_hints["interaction_policy_association_reweighting_reason"] = interaction_policy_packet["association_reweighting_reason"]
        expression_hints["interaction_policy_insight_terrain_shape_target"] = interaction_policy_packet["insight_terrain_shape_target"]
        expression_hints["interaction_policy_overnight_bias_roles"] = interaction_policy_packet["overnight_bias_roles"]
        expression_hints["interaction_policy_reaction_vs_overnight_bias"] = interaction_policy_packet["reaction_vs_overnight_bias"]
        expression_hints["surface_voice_texture"] = str((interaction_policy_packet.get("expressive_style_state") or {}).get("state") or "grounded_gentle")
        expression_hints["surface_lightness_room"] = round(float((interaction_policy_packet.get("expressive_style_state") or {}).get("lightness_room") or 0.0), 4)
        expression_hints["surface_continuity_weight"] = round(float((interaction_policy_packet.get("expressive_style_state") or {}).get("continuity_weight") or 0.0), 4)
        expression_hints["surface_relational_voice_texture"] = str((interaction_policy_packet.get("relational_style_memory_state") or {}).get("state") or "grounded_gentle")
        expression_hints["surface_relational_playful_ceiling"] = round(float((interaction_policy_packet.get("relational_style_memory_state") or {}).get("playful_ceiling") or 0.0), 4)
        expression_hints["surface_cultural_register"] = str((interaction_policy_packet.get("cultural_conversation_state") or {}).get("state") or "careful_polite")
        expression_hints["surface_cultural_joke_ratio_ceiling"] = round(float((interaction_policy_packet.get("cultural_conversation_state") or {}).get("joke_ratio_ceiling") or 0.0), 4)
        expression_hints["surface_lightness_budget_state"] = str((interaction_policy_packet.get("lightness_budget_state") or {}).get("state") or "grounded_only")
        expression_hints["surface_lightness_banter_room"] = round(float((interaction_policy_packet.get("lightness_budget_state") or {}).get("banter_room") or 0.0), 4)
        expression_hints["surface_learning_mode_state"] = str((interaction_policy_packet.get("learning_mode_state") or {}).get("state") or "observe_only")
        expression_hints["surface_social_experiment_state"] = str((interaction_policy_packet.get("social_experiment_loop_state") or {}).get("state") or "watch_and_read")
        expression_hints["surface_identity_arc_kind"] = str(interaction_policy_packet.get("identity_arc_kind") or "")
        expression_hints["surface_identity_arc_phase"] = str(interaction_policy_packet.get("identity_arc_phase") or "")
        expression_hints["surface_identity_arc_open_tension"] = str(interaction_policy_packet.get("identity_arc_open_tension") or "")
        expression_hints["action_posture"] = action_posture
        expression_hints["action_posture_mode"] = action_posture["engagement_mode"]
        expression_hints["action_posture_goal"] = action_posture["outcome_goal"]
        expression_hints["action_posture_boundary"] = action_posture["boundary_mode"]
        expression_hints["actuation_plan"] = actuation_plan
        expression_hints["actuation_execution_mode"] = actuation_plan["execution_mode"]
        expression_hints["actuation_primary_action"] = actuation_plan["primary_action"]
        expression_hints["actuation_reply_permission"] = actuation_plan["reply_permission"]
        expression_hints["actuation_wait_before_action"] = actuation_plan["wait_before_action"]
        expression_hints["long_term_theme_focus"] = long_term_theme_focus
        expression_hints["long_term_theme_anchor"] = long_term_theme_anchor
        expression_hints["replay_intensity"] = round(replay_intensity, 4)
        expression_hints["anticipation_tension"] = round(anticipation_tension, 4)
        expression_hints["stabilization_drive"] = round(stabilization_drive, 4)
        expression_hints["relational_clarity"] = round(relational_clarity, 4)
        expression_hints["meaning_inertia"] = round(meaning_inertia, 4)
        expression_hints["roughness_level"] = round(roughness_level, 4)
        expression_hints["roughness_velocity"] = round(roughness_velocity, 4)
        expression_hints["roughness_momentum"] = round(roughness_momentum, 4)
        expression_hints["roughness_dwell"] = round(roughness_dwell, 4)
        expression_hints["defensive_level"] = round(defensive_level, 4)
        expression_hints["defensive_velocity"] = round(defensive_velocity, 4)
        expression_hints["defensive_momentum"] = round(defensive_momentum, 4)
        expression_hints["defensive_dwell"] = round(defensive_dwell, 4)
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
        counterpart_person_id = _text_or_none(
            world_snapshot.get("person_id")
            or current_state.get("related_person_id")
        )
        identity_trace = self.memory_core.load_latest_identity_trace(
            culture_id=str(world_snapshot.get("culture_id") or "") or None,
            community_id=str(world_snapshot.get("community_id") or "") or None,
            social_role=str(world_snapshot.get("social_role") or "") or None,
            related_person_id=counterpart_person_id,
        )
        relationship_trace = self.memory_core.load_latest_profile_record(
            kind="relationship_trace",
            culture_id=str(world_snapshot.get("culture_id") or "") or None,
            community_id=str(world_snapshot.get("community_id") or "") or None,
            social_role=str(world_snapshot.get("social_role") or "") or None,
            memory_anchor=str(world_snapshot.get("place_memory_anchor") or "") or None,
            related_person_id=counterpart_person_id,
        )
        community_profile_trace = self.memory_core.load_latest_profile_record(
            kind="community_profile_trace",
            culture_id=str(world_snapshot.get("culture_id") or "") or None,
            community_id=str(world_snapshot.get("community_id") or "") or None,
        )
        context_shift_trace = self.memory_core.load_latest_profile_record(
            kind="context_shift_trace",
            culture_id=str(world_snapshot.get("culture_id") or "") or None,
            community_id=str(world_snapshot.get("community_id") or "") or None,
        )
        working_memory_trace = self.memory_core.load_latest_profile_record(
            kind="working_memory_trace",
            culture_id=str(world_snapshot.get("culture_id") or "") or None,
            community_id=str(world_snapshot.get("community_id") or "") or None,
            social_role=str(world_snapshot.get("social_role") or "") or None,
        )
        merged_state = dict(identity_trace or {})
        merged_state.update(dict(relationship_trace or {}))
        merged_state.update(dict(community_profile_trace or {}))
        merged_state.update(dict(context_shift_trace or {}))
        merged_state.update(dict(working_memory_trace or {}))
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
        field_estimate = self.field_estimator_core.snapshot(
            current_state=current_state,
            observed_roughness=_float_from(current_state, "terrain_transition_roughness", default=0.0),
            observed_defensive_salience=_float_from(current_state, "defensive_salience", default=0.0),
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
            terrain_transition_roughness=field_estimate.roughness_level,
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
            terrain_transition_roughness=field_estimate.roughness_level,
            caution_bias=personality.caution_bias,
            recent_strain=_float_from(merged_state, "recent_strain", default=0.32),
            continuity_score=_float_from(merged_state, "continuity_score", default=0.48),
            defensive_salience=field_estimate.defensive_level,
        )
        merged_state["terrain_transition_roughness"] = field_estimate.roughness_level
        merged_state["roughness_level"] = field_estimate.roughness_level
        merged_state["roughness_velocity"] = field_estimate.roughness_velocity
        merged_state["roughness_momentum"] = field_estimate.roughness_momentum
        merged_state["roughness_dwell"] = field_estimate.roughness_dwell
        merged_state["defensive_salience"] = field_estimate.defensive_level
        merged_state["defensive_level"] = field_estimate.defensive_level
        merged_state["defensive_velocity"] = field_estimate.defensive_velocity
        merged_state["defensive_momentum"] = field_estimate.defensive_momentum
        merged_state["defensive_dwell"] = field_estimate.defensive_dwell
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
            terrain_transition_roughness=field_estimate.roughness_level,
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
        working_memory = self.working_memory_core.snapshot(
            user_input=user_input,
            sensor_input=current_state,
            current_state=merged_state,
            relational_world=world_snapshot,
            previous_trace=working_memory_trace,
            recall_payload=recall_payload,
        )
        settled_working_memory = self.working_memory_core.settle_after_turn(
            snapshot=working_memory,
            reply_text=reply_text,
            current_state=current_state,
            recall_payload=recall_payload,
        )
        candidate_signal = _memory_candidate_influence(memory_write_candidates or [])
        interaction_alignment = _interaction_alignment_snapshot(
            current_state=current_state,
            output=output,
        )
        adjusted_pending_meaning = _clamp01(
            settled_working_memory.pending_meaning
            + candidate_signal["meaning_pull"] * 0.18
            + candidate_signal["continuity_pull"] * 0.06
            + interaction_alignment["alignment_score"] * 0.04
        )
        adjusted_memory_pressure = _clamp01(
            settled_working_memory.memory_pressure
            + candidate_signal["meaning_pull"] * 0.12
            + candidate_signal["continuity_pull"] * 0.08
            + candidate_signal["social_pull"] * 0.06
            + interaction_alignment["mismatch_intensity"] * 0.06
        )
        adjusted_focus = settled_working_memory.current_focus
        if candidate_signal["focus_hint"] and adjusted_focus == "ambient":
            adjusted_focus = candidate_signal["focus_hint"]
        adjusted_focus_anchor = settled_working_memory.focus_anchor or candidate_signal["anchor_hint"]
        settled_working_memory = replace(
            settled_working_memory,
            focus_anchor=adjusted_focus_anchor[:160],
            current_focus=adjusted_focus,
            pending_meaning=round(adjusted_pending_meaning, 4),
            memory_pressure=round(adjusted_memory_pressure, 4),
        )
        replay_signature_focus = _text_or_none((recall_payload or {}).get("replay_signature_focus")) or _text_or_none(current_state.get("working_memory_replay_focus"))
        replay_signature_anchor = _text_or_none((recall_payload or {}).get("replay_signature_anchor")) or _text_or_none(current_state.get("working_memory_replay_anchor"))
        replay_signature_strength = _float_from(
            recall_payload,
            "replay_signature_strength",
            default=_float_from(current_state, "working_memory_replay_strength", default=0.0),
        )
        replay_signature_alignment = _replay_signature_alignment(
            recall_payload=recall_payload or {},
            current_state=current_state,
            focus=replay_signature_focus,
            anchor=replay_signature_anchor,
        )
        replay_signature_reinforcement = _clamp01(replay_signature_alignment * replay_signature_strength)
        long_term_theme_summary = _text_or_none((recall_payload or {}).get("long_term_theme_summary")) or _text_or_none(current_state.get("long_term_theme_summary"))
        long_term_theme_alignment = _long_term_theme_alignment(
            recall_payload=recall_payload or {},
            reply_text=reply_text,
            summary=long_term_theme_summary,
        )
        long_term_theme_strength = _float_from(
            recall_payload,
            "long_term_theme_strength",
            default=_float_from(current_state, "long_term_theme_strength", default=0.0),
        )
        long_term_theme_reinforcement = _clamp01(long_term_theme_alignment * long_term_theme_strength)
        memory_orchestration["consolidation_priority"] = _clamp01(
            _float_from(memory_orchestration, "consolidation_priority", default=0.0)
            + replay_signature_reinforcement * 0.12
            + long_term_theme_reinforcement * 0.08
            + candidate_signal["meaning_pull"] * 0.08
            + candidate_signal["continuity_pull"] * 0.05
            + interaction_alignment["alignment_score"] * 0.05
        )
        continuity_with_candidates = _clamp01(
            persistence.continuity_score
            + candidate_signal["continuity_pull"] * 0.04
            + interaction_alignment["alignment_score"] * 0.03
            - interaction_alignment["mismatch_intensity"] * 0.02
        )
        social_grounding_with_candidates = _clamp01(
            persistence.social_grounding
            + candidate_signal["social_pull"] * 0.04
            + interaction_alignment["alignment_score"] * 0.04
            - interaction_alignment["mismatch_intensity"] * 0.03
        )
        recent_strain_with_candidates = _clamp01(
            max(
                0.0,
                persistence.recent_strain
                - candidate_signal["continuity_pull"] * 0.03
                + interaction_alignment["mismatch_intensity"] * 0.05
            )
        )
        attachment_with_candidates = _clamp01(
            relationship.attachment
            + candidate_signal["social_pull"] * 0.03
            + interaction_alignment["alignment_score"] * 0.02
        )
        trust_memory_with_candidates = _clamp01(
            relationship.trust_memory
            + candidate_signal["social_pull"] * 0.02
            + interaction_alignment["alignment_score"] * 0.03
            - interaction_alignment["mismatch_intensity"] * 0.02
        )
        familiarity_with_candidates = _clamp01(
            relationship.familiarity
            + candidate_signal["continuity_pull"] * 0.02
            + interaction_alignment["alignment_score"] * 0.02
        )
        raw_core_axes = _core_state_axes(
            stress=lingering,
            recovery_need=max(0.0, recovery_need * 0.92),
            temporal_pressure=temporal_pressure,
            continuity_score=continuity_with_candidates,
            social_grounding=social_grounding_with_candidates,
            recent_strain=recent_strain_with_candidates,
            trust_bias=development.trust_bias,
            terrain_transition_roughness=field_estimate.roughness_level,
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
            continuity_score=continuity_with_candidates,
            social_grounding=social_grounding_with_candidates,
            recall_active=bool(recall_payload),
            interaction_afterglow=interaction_afterglow,
        )
        next_related_person_ids = collect_related_person_ids(
            candidate_signal["target_person_id"],
            counterpart_person_id,
            current_state.get("related_person_id"),
            current_state.get("related_person_ids"),
            registry_snapshot=current_state.get("person_registry_snapshot"),
            limit=4,
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
            attachment=attachment_with_candidates,
            trust_memory=trust_memory_with_candidates,
            familiarity=familiarity_with_candidates,
            role_alignment=relationship.role_alignment,
            rupture_sensitivity=relationship.rupture_sensitivity,
            caution_bias=personality.caution_bias,
            affiliation_bias=personality.affiliation_bias,
            exploration_bias=personality.exploration_bias,
            reflective_bias=personality.reflective_bias,
            continuity_score=continuity_with_candidates,
            social_grounding=social_grounding_with_candidates,
            recent_strain=recent_strain_with_candidates,
            culture_resonance=persistence.culture_resonance,
            community_resonance=persistence.community_resonance,
            terrain_transition_roughness=field_estimate.roughness_level,
            roughness_level=field_estimate.roughness_level,
            roughness_velocity=field_estimate.roughness_velocity,
            roughness_momentum=field_estimate.roughness_momentum,
            roughness_dwell=field_estimate.roughness_dwell,
            near_body_risk=_float_from(current_state, "near_body_risk", default=0.0),
            defensive_salience=field_estimate.defensive_level,
            defensive_level=field_estimate.defensive_level,
            defensive_velocity=field_estimate.defensive_velocity,
            defensive_momentum=field_estimate.defensive_momentum,
            defensive_dwell=field_estimate.defensive_dwell,
            approach_confidence=_float_from(current_state, "approach_confidence", default=0.0),
            reachability=_float_from(current_state, "reachability", default=0.0),
            reuse_trajectory=memory_orchestration["reuse_trajectory"],
            interference_pressure=memory_orchestration["interference_pressure"],
            consolidation_priority=memory_orchestration["consolidation_priority"],
            prospective_memory_pull=memory_orchestration["prospective_memory_pull"],
            working_memory_pressure=settled_working_memory.memory_pressure,
            unresolved_count=settled_working_memory.unresolved_count,
            current_focus=settled_working_memory.current_focus,
            pending_meaning=settled_working_memory.pending_meaning,
            related_person_id=candidate_signal["target_person_id"] or counterpart_person_id or "",
            related_person_ids=next_related_person_ids,
            relation_seed_summary=_text_or_none(current_state.get("relation_seed_summary")),
            partner_address_hint=_text_or_none(current_state.get("partner_address_hint")),
            partner_timing_hint=_text_or_none(current_state.get("partner_timing_hint")),
            partner_stance_hint=_text_or_none(current_state.get("partner_stance_hint")),
            partner_social_interpretation=_text_or_none(current_state.get("partner_social_interpretation")),
            conscious_residue_focus=settled_working_memory.conscious_residue_focus,
            conscious_residue_anchor=settled_working_memory.conscious_residue_anchor,
            conscious_residue_summary=settled_working_memory.conscious_residue_summary,
            conscious_residue_strength=settled_working_memory.conscious_residue_strength,
            autobiographical_thread_mode=settled_working_memory.autobiographical_thread_mode,
            autobiographical_thread_anchor=settled_working_memory.autobiographical_thread_anchor,
            autobiographical_thread_focus=settled_working_memory.autobiographical_thread_focus,
            autobiographical_thread_strength=settled_working_memory.autobiographical_thread_strength,
            autobiographical_thread_reasons=list(settled_working_memory.autobiographical_thread_reasons),
            long_term_theme_focus=settled_working_memory.long_term_theme_focus,
            long_term_theme_anchor=settled_working_memory.long_term_theme_anchor,
            long_term_theme_kind=settled_working_memory.long_term_theme_kind,
            long_term_theme_summary=settled_working_memory.long_term_theme_summary,
            long_term_theme_strength=settled_working_memory.long_term_theme_strength,
            interaction_alignment_score=interaction_alignment["alignment_score"],
            shared_attention_delta=interaction_alignment["shared_attention_delta"],
            distance_mismatch=interaction_alignment["distance_mismatch"],
            hesitation_mismatch=interaction_alignment["hesitation_mismatch"],
            opening_pace_mismatch=interaction_alignment["opening_pace_mismatch"],
            return_gaze_mismatch=interaction_alignment["return_gaze_mismatch"],
        )
        terrain_position_state = _coerce_affective_position_state(
            current_state.get("prev_affective_position") or current_state.get("affective_position"),
            default_dim=_default_affective_position_dim(current_state.get("qualia_axis_labels")),
        )
        terrain_state_before = _coerce_affective_terrain_state(
            current_state.get("affective_terrain_state"),
            position_dim=terrain_position_state.position_dim,
        )
        terrain_readout_before = _coerce_terrain_readout(
            current_state.get("terrain_readout"),
            terrain_state=terrain_state_before,
            position_state=terrain_position_state,
        )
        protection_mode_payload = _mapping_or_empty(current_state.get("protection_mode"))
        association_graph_state_before = coerce_association_graph_state(
            current_state.get("association_graph_state")
        )
        insight_event_payload = _mapping_or_empty(current_state.get("insight_event"))
        terrain_plasticity_update: Optional[TerrainPlasticityUpdate] = None
        terrain_plasticity_signals: Dict[str, float] = {}
        terrain_state_after = terrain_state_before
        if _has_affective_terrain_context(current_state):
            terrain_plasticity_signals = _derive_terrain_plasticity_signals(
                next_state=next_state,
                interaction_alignment=interaction_alignment,
                candidate_signal=candidate_signal,
                terrain_readout=terrain_readout_before,
                protection_mode=protection_mode_payload,
                reply_present=bool(reply_text),
            )
            terrain_plasticity_update = derive_terrain_plasticity_update(
                position_state=terrain_position_state,
                terrain_readout=terrain_readout_before,
                safety_gain=terrain_plasticity_signals["safety_gain"],
                strain_load=terrain_plasticity_signals["strain_load"],
                bond_weight=terrain_plasticity_signals["bond_weight"],
                unresolved_tension=terrain_plasticity_signals["unresolved_tension"],
                terrain_reweighting_bias=_float_from(current_state, "terrain_reweighting_bias", default=0.0),
                memory_class_focus=_text_or_none(current_state.get("memory_write_class")) or "episodic",
                commitment_carry_bias=_float_from(current_state, "commitment_carry_bias", default=0.0),
                commitment_target_focus=_text_or_none(current_state.get("commitment_target_focus")) or "",
                commitment_state_focus=_text_or_none(current_state.get("commitment_state_focus")) or "",
                insight_terrain_shape_bias=_float_from(current_state, "insight_terrain_shape_bias", default=0.0),
                insight_terrain_shape_reason=_text_or_none(current_state.get("insight_terrain_shape_reason")) or "",
                insight_anchor_center=current_state.get("insight_anchor_center"),
                insight_anchor_dispersion=_float_from(current_state, "insight_anchor_dispersion", default=0.0),
                qualia_body_load=_float_from(
                    _mapping_or_empty(current_state.get("qualia_planner_view")),
                    "body_load",
                    default=0.0,
                ),
                qualia_degraded=bool(_mapping_or_empty(current_state.get("qualia_planner_view")).get("degraded", False)),
                protection_mode_name=str(protection_mode_payload.get("mode") or ""),
                dt=1.0,
            )
            terrain_state_after = apply_terrain_plasticity(
                terrain_state_before,
                terrain_plasticity_update,
            )
        association_graph_state_after = apply_association_reinforcement(
            association_graph_state_before,
            insight_event_payload,
            association_reweighting_focus=_text_or_none(current_state.get("association_reweighting_focus")) or "",
            association_reweighting_reason=_text_or_none(current_state.get("association_reweighting_reason")) or "",
            commitment_followup_focus=_text_or_none(current_state.get("commitment_followup_focus")) or "",
            commitment_carry_bias=_float_from(current_state, "commitment_carry_bias", default=0.0),
        )
        insight_trace = derive_insight_trace(
            insight_event=insight_event_payload,
            qualia_planner_view=_mapping_or_empty(current_state.get("qualia_planner_view")),
            protection_mode=protection_mode_payload,
            affective_position=terrain_position_state.to_dict(),
            terrain_readout=terrain_readout_before.to_dict(),
        )
        initiative_followup_bias = _derive_initiative_followup_bias(
            body_recovery_guard=_mapping_or_empty(current_state.get("body_recovery_guard")),
            initiative_readiness=_mapping_or_empty(current_state.get("initiative_readiness")),
            commitment_state=_mapping_or_empty(current_state.get("commitment_state")),
            protection_mode=protection_mode_payload,
            temperament_estimate=derive_temperament_estimate(current_state).to_dict(),
            memory_write_class=_text_or_none(current_state.get("memory_write_class")) or "episodic",
            pending_meaning=settled_working_memory.pending_meaning,
            prospective_memory_pull=memory_orchestration["prospective_memory_pull"],
            reply_present=bool(reply_text),
        )
        temperament_trace = advance_temperament_traces(
            current_state,
            protection_mode=str(protection_mode_payload.get("mode") or ""),
            protection_strength=_float_from(protection_mode_payload, "strength", default=0.0),
            body_recovery_guard=str((_mapping_or_empty(current_state.get("body_recovery_guard"))).get("state") or "open"),
            initiative_readiness=str((_mapping_or_empty(current_state.get("initiative_readiness"))).get("state") or "hold"),
            initiative_followup_state=str(initiative_followup_bias.get("state") or "hold"),
            memory_write_class=_text_or_none(current_state.get("memory_write_class")) or "episodic",
        )
        memory_write_class = _text_or_none(current_state.get("memory_write_class")) or "episodic"
        memory_write_class_reason = _text_or_none(current_state.get("memory_write_class_reason")) or "default_episode"
        appends = [dict(item) for item in (memory_write_candidates or [])]
        if reconstructed:
            reconstructed["working_memory_replay_focus"] = replay_signature_focus
            reconstructed["working_memory_replay_anchor"] = replay_signature_anchor
            reconstructed["working_memory_replay_strength"] = round(replay_signature_strength, 4)
            reconstructed["working_memory_replay_alignment"] = round(replay_signature_alignment, 4)
            reconstructed["working_memory_replay_reinforcement"] = round(replay_signature_reinforcement, 4)
            reconstructed["long_term_theme_summary"] = long_term_theme_summary
            reconstructed["long_term_theme_alignment"] = round(long_term_theme_alignment, 4)
            reconstructed["long_term_theme_reinforcement"] = round(long_term_theme_reinforcement, 4)
            reconstructed["consolidation_priority"] = round(memory_orchestration["consolidation_priority"], 4)
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
                "related_person_id": candidate_signal["target_person_id"] or None,
                "attachment": round(next_state.attachment, 4),
                "trust_memory": round(next_state.trust_memory, 4),
                "familiarity": round(next_state.familiarity, 4),
                "role_alignment": round(next_state.role_alignment, 4),
                "rupture_sensitivity": round(next_state.rupture_sensitivity, 4),
                "caution_bias": round(next_state.caution_bias, 4),
                "affiliation_bias": round(next_state.affiliation_bias, 4),
                "reuse_trajectory": round(next_state.reuse_trajectory, 4),
                "consolidation_priority": round(next_state.consolidation_priority, 4),
                "candidate_social_pull": round(candidate_signal["social_pull"], 4),
                "interaction_alignment_score": round(next_state.interaction_alignment_score, 4),
                "shared_attention_delta": round(next_state.shared_attention_delta, 4),
                "distance_mismatch": round(next_state.distance_mismatch, 4),
                "hesitation_mismatch": round(next_state.hesitation_mismatch, 4),
                "opening_pace_mismatch": round(next_state.opening_pace_mismatch, 4),
                "return_gaze_mismatch": round(next_state.return_gaze_mismatch, 4),
                "address_hint": next_state.partner_address_hint,
                "timing_hint": next_state.partner_timing_hint,
                "stance_hint": next_state.partner_stance_hint,
                "social_interpretation": next_state.partner_social_interpretation,
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
                "related_person_id": candidate_signal["target_person_id"] or None,
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
                "candidate_continuity_pull": round(candidate_signal["continuity_pull"], 4),
                "candidate_meaning_pull": round(candidate_signal["meaning_pull"], 4),
                "candidate_social_pull": round(candidate_signal["social_pull"], 4),
                "candidate_target_person_id": candidate_signal["target_person_id"],
                "interaction_alignment_score": round(next_state.interaction_alignment_score, 4),
                "shared_attention_delta": round(next_state.shared_attention_delta, 4),
                "distance_mismatch": round(next_state.distance_mismatch, 4),
                "hesitation_mismatch": round(next_state.hesitation_mismatch, 4),
                "opening_pace_mismatch": round(next_state.opening_pace_mismatch, 4),
                "return_gaze_mismatch": round(next_state.return_gaze_mismatch, 4),
            }
        )
        if transition_signal["transition_intensity"] > 0.0 or _float_from(current_state, "terrain_transition_roughness", default=0.0) > 0.0:
            appends.append(
                {
                    "kind": "context_shift_trace",
                    "summary": "slow context shift trace",
                    "text": "slow context shift trace",
                    "memory_anchor": world_snapshot.get("place_memory_anchor") or _text_or_none(current_state.get("memory_anchor")) or "context-shift",
                    "culture_id": world_snapshot.get("culture_id"),
                    "community_id": world_snapshot.get("community_id"),
                    "social_role": world_snapshot.get("social_role"),
                    "transition_intensity": round(_float_from(transition_signal, "transition_intensity", default=0.0), 4),
                    "place_changed": bool(transition_signal.get("place_changed")),
                    "body_state_changed": bool(transition_signal.get("body_state_changed")),
                    "privacy_shift": bool(transition_signal.get("privacy_shift")),
                    "density_shift": bool(transition_signal.get("density_shift")),
                    "terrain_transition_roughness": round(_float_from(current_state, "terrain_transition_roughness", default=0.0), 4),
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
                "terrain_transition_roughness": round(next_state.terrain_transition_roughness, 4),
                "roughness_level": round(next_state.roughness_level, 4),
                "roughness_dwell": round(next_state.roughness_dwell, 4),
                "defensive_level": round(next_state.defensive_level, 4),
                "defensive_dwell": round(next_state.defensive_dwell, 4),
                "social_update_strength": round(development.social_update_strength, 4),
                "identity_update_strength": round(development.identity_update_strength, 4),
                "prospective_memory_pull": round(next_state.prospective_memory_pull, 4),
                "interference_pressure": round(next_state.interference_pressure, 4),
                "candidate_continuity_pull": round(candidate_signal["continuity_pull"], 4),
                "candidate_meaning_pull": round(candidate_signal["meaning_pull"], 4),
            }
        )
        appends.append(
            self.working_memory_core.build_trace_record(
                snapshot=settled_working_memory,
                current_state=current_state,
                relational_world=world_snapshot,
            )
        )
        commitment_state_payload = _mapping_or_empty(current_state.get("commitment_state"))
        agenda_state_payload = _mapping_or_empty(current_state.get("agenda_state"))
        commitment_target = _text_or_none(commitment_state_payload.get("target")) or ""
        commitment_mode = _text_or_none(commitment_state_payload.get("state")) or "waver"
        commitment_score = _float_from(commitment_state_payload, "score", default=0.0)
        commitment_margin = _float_from(commitment_state_payload, "winner_margin", default=0.0)
        commitment_cost = _float_from(commitment_state_payload, "accepted_cost", default=0.0)
        agenda_mode = _text_or_none(agenda_state_payload.get("state")) or "hold"
        agenda_reason = _text_or_none(agenda_state_payload.get("reason")) or ""
        agenda_score = _float_from(agenda_state_payload, "score", default=0.0)
        agenda_margin = _float_from(agenda_state_payload, "winner_margin", default=0.0)
        if commitment_target and (commitment_mode != "waver" or commitment_score >= 0.2):
            appends.append(
                {
                    "kind": "commitment_trace",
                    "summary": f"commitment:{commitment_mode}:{commitment_target}",
                    "text": f"commitment:{commitment_mode}:{commitment_target}",
                    "memory_anchor": _text_or_none(current_state.get("memory_anchor")) or commitment_target,
                    "culture_id": world_snapshot.get("culture_id"),
                    "community_id": world_snapshot.get("community_id"),
                    "social_role": world_snapshot.get("social_role"),
                    "commitment_state": commitment_mode,
                    "commitment_target": commitment_target,
                    "commitment_score": round(commitment_score, 4),
                    "commitment_winner_margin": round(commitment_margin, 4),
                    "commitment_accepted_cost": round(commitment_cost, 4),
                }
            )
        if agenda_mode and (agenda_mode != "hold" or agenda_score >= 0.18):
            appends.append(
                {
                    "kind": "agenda_trace",
                    "summary": f"agenda:{agenda_mode}:{agenda_reason or 'none'}",
                    "text": f"agenda:{agenda_mode}:{agenda_reason or 'none'}",
                    "memory_anchor": _text_or_none(current_state.get("memory_anchor")) or agenda_reason or agenda_mode,
                    "culture_id": world_snapshot.get("culture_id"),
                    "community_id": world_snapshot.get("community_id"),
                    "social_role": world_snapshot.get("social_role"),
                    "related_person_id": candidate_signal["target_person_id"] or counterpart_person_id,
                    "group_thread_id": _text_or_none(current_state.get("group_thread_id")) or "",
                    "agenda_state": agenda_mode,
                    "agenda_reason": agenda_reason,
                    "agenda_score": round(agenda_score, 4),
                    "agenda_winner_margin": round(agenda_margin, 4),
                    "commitment_state": commitment_mode,
                    "commitment_target": commitment_target,
                }
            )
        if insight_trace is not None:
            appends.append(
                {
                    "kind": "insight_trace",
                    "summary": insight_trace.reframed_topic or "insight trace",
                    "text": insight_trace.reframed_topic or "insight trace",
                    "memory_anchor": _text_or_none(current_state.get("memory_anchor")) or insight_trace.reframed_topic or "insight-trace",
                    "culture_id": world_snapshot.get("culture_id"),
                    "community_id": world_snapshot.get("community_id"),
                    "social_role": world_snapshot.get("social_role"),
                    **insight_trace.to_dict(),
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
        interaction_policy_packet = (
            dict(current_state.get("interaction_policy_packet") or {})
            if isinstance(current_state.get("interaction_policy_packet"), Mapping)
            else {}
        )
        social_topology_state = dict(
            interaction_policy_packet.get("social_topology_state")
            or current_state.get("social_topology_state")
            or {}
        )
        relation_competition_state = dict(
            interaction_policy_packet.get("relation_competition_state")
            or current_state.get("relation_competition_state")
            or {}
        )
        group_thread_focus = str(social_topology_state.get("state") or "").strip()
        group_top_person_ids = [
            str(item).strip()
            for item in list(relation_competition_state.get("top_person_ids") or [])
            if str(item).strip()
        ]
        group_dominant_person_id = str(
            relation_competition_state.get("dominant_person_id")
            or candidate_signal["target_person_id"]
            or counterpart_person_id
            or ""
        ).strip()
        recent_dialogue_state = dict(
            interaction_policy_packet.get("recent_dialogue_state")
            or current_state.get("recent_dialogue_state")
            or {}
        )
        discussion_thread_state = dict(
            interaction_policy_packet.get("discussion_thread_state")
            or current_state.get("discussion_thread_state")
            or {}
        )
        issue_state = dict(
            interaction_policy_packet.get("issue_state")
            or current_state.get("issue_state")
            or {}
        )
        if group_thread_focus:
            appends.append(
                {
                    "kind": "group_thread_trace",
                    "source": "post_turn",
                    "summary": f"{group_thread_focus}:{group_dominant_person_id or 'ambient'}",
                    "text": reply_text or str(user_input.get("text") or "").strip(),
                    "memory_anchor": _text_or_none(current_state.get("memory_anchor")) or group_thread_focus,
                    "group_thread_id": str(current_state.get("group_thread_id") or "").strip(),
                    "group_thread_focus": group_thread_focus,
                    "related_person_id": group_dominant_person_id,
                    "top_person_ids": group_top_person_ids,
                    "thread_total_people": int(
                        relation_competition_state.get("total_people")
                        or social_topology_state.get("total_people")
                        or len(group_top_person_ids)
                        or 0
                    ),
                    "threading_pressure": round(float(social_topology_state.get("threading_pressure") or 0.0), 4),
                    "visibility_pressure": round(float(social_topology_state.get("visibility_pressure") or 0.0), 4),
                    "hierarchy_pressure": round(float(social_topology_state.get("hierarchy_pressure") or 0.0), 4),
                    "continuity_score": round(next_state.continuity_score, 4),
                    "social_grounding": round(next_state.social_grounding, 4),
                    "culture_id": world_snapshot.get("culture_id"),
                    "community_id": world_snapshot.get("community_id"),
                    "social_role": world_snapshot.get("social_role"),
                }
            )
        discussion_anchor = str(
            issue_state.get("issue_anchor")
            or discussion_thread_state.get("topic_anchor")
            or recent_dialogue_state.get("recent_anchor")
            or ""
        ).strip()
        discussion_state_name = str(discussion_thread_state.get("state") or "").strip()
        issue_state_name = str(issue_state.get("state") or "").strip()
        if discussion_anchor or discussion_state_name or issue_state_name:
            appends.append(
                {
                    "kind": "discussion_thread_trace",
                    "source": "post_turn",
                    "summary": f"{discussion_anchor or 'ambient'}:{issue_state_name or discussion_state_name or 'ambient'}",
                    "text": reply_text or str(user_input.get("text") or "").strip(),
                    "memory_anchor": discussion_anchor or _text_or_none(current_state.get("memory_anchor")),
                    "recent_dialogue_state": str(recent_dialogue_state.get("state") or "").strip(),
                    "recent_dialogue_overlap": round(float(recent_dialogue_state.get("overlap_score") or 0.0), 4),
                    "recent_dialogue_reopen_pressure": round(float(recent_dialogue_state.get("reopen_pressure") or 0.0), 4),
                    "recent_dialogue_thread_carry": round(float(recent_dialogue_state.get("thread_carry") or 0.0), 4),
                    "discussion_thread_state": discussion_state_name,
                    "discussion_thread_anchor": str(discussion_thread_state.get("topic_anchor") or "").strip(),
                    "discussion_unresolved_pressure": round(float(discussion_thread_state.get("unresolved_pressure") or 0.0), 4),
                    "discussion_revisit_readiness": round(float(discussion_thread_state.get("revisit_readiness") or 0.0), 4),
                    "discussion_thread_visibility": round(float(discussion_thread_state.get("thread_visibility") or 0.0), 4),
                    "issue_state": issue_state_name,
                    "issue_anchor": str(issue_state.get("issue_anchor") or "").strip(),
                    "issue_question_pressure": round(float(issue_state.get("question_pressure") or 0.0), 4),
                    "issue_pause_readiness": round(float(issue_state.get("pause_readiness") or 0.0), 4),
                    "issue_resolution_readiness": round(float(issue_state.get("resolution_readiness") or 0.0), 4),
                }
            )
        for append in appends:
            append.setdefault("memory_write_class", memory_write_class)
            append.setdefault("memory_write_class_reason", memory_write_class_reason)
            append.setdefault("followup_bias", round(float(initiative_followup_bias.get("score", 0.0) or 0.0), 4))
            append.setdefault("commitment_state", commitment_mode)
            append.setdefault("commitment_target", commitment_target)
            append.setdefault("commitment_score", round(commitment_score, 4))
            append.setdefault("commitment_winner_margin", round(commitment_margin, 4))
            append.setdefault("commitment_accepted_cost", round(commitment_cost, 4))
        stored = self.memory_core.append_records(appends)
        registry_style_traits = _derive_registry_style_traits(
            current_state=current_state,
            next_state=next_state,
            reply_present=bool(reply_text),
        )
        person_registry_snapshot = _updated_person_registry_snapshot(
            existing_snapshot=_mapping_or_empty(current_state.get("person_registry_snapshot"))
            if isinstance(current_state, Mapping)
            else {},
            target_person_id=candidate_signal["target_person_id"] or counterpart_person_id,
            attachment=next_state.attachment,
            familiarity=next_state.familiarity,
            trust_memory=next_state.trust_memory,
            continuity_score=next_state.continuity_score,
            social_grounding=next_state.social_grounding,
            current_focus=next_state.current_focus,
            community_id=str(world_snapshot.get("community_id") or ""),
            culture_id=str(world_snapshot.get("culture_id") or ""),
            social_role=str(world_snapshot.get("social_role") or ""),
            style_warmth_memory=registry_style_traits["style_warmth_memory"],
            playful_ceiling=registry_style_traits["playful_ceiling"],
            advice_tolerance=registry_style_traits["advice_tolerance"],
            lexical_familiarity=registry_style_traits["lexical_familiarity"],
            lexical_variation_bias=registry_style_traits["lexical_variation_bias"],
        )
        group_thread_registry_snapshot = _updated_group_thread_registry_snapshot(
            existing_snapshot=_mapping_or_empty(current_state.get("group_thread_registry_snapshot"))
            if isinstance(current_state, Mapping)
            else {},
            thread_hint=_text_or_none(current_state.get("group_thread_id")) or "",
            topology_state=social_topology_state,
            dominant_person_id=group_dominant_person_id,
            top_person_ids=group_top_person_ids,
            total_people=int(
                relation_competition_state.get("total_people")
                or social_topology_state.get("total_people")
                or len(group_top_person_ids)
                or 0
            ),
            continuity_score=next_state.continuity_score,
            social_grounding=next_state.social_grounding,
            community_id=str(world_snapshot.get("community_id") or ""),
            culture_id=str(world_snapshot.get("culture_id") or ""),
            social_role=str(world_snapshot.get("social_role") or ""),
        )
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
                "roughness_level": round(field_estimate.roughness_level, 4),
                "roughness_velocity": round(field_estimate.roughness_velocity, 4),
                "roughness_momentum": round(field_estimate.roughness_momentum, 4),
                "roughness_dwell": round(field_estimate.roughness_dwell, 4),
                "defensive_level": round(field_estimate.defensive_level, 4),
                "defensive_velocity": round(field_estimate.defensive_velocity, 4),
                "defensive_momentum": round(field_estimate.defensive_momentum, 4),
                "defensive_dwell": round(field_estimate.defensive_dwell, 4),
                "recovery_reopening": round(core_axes["recovery_reopening"], 4),
                "reuse_trajectory": round(memory_orchestration["reuse_trajectory"], 4),
                "interference_pressure": round(memory_orchestration["interference_pressure"], 4),
                "consolidation_priority": round(memory_orchestration["consolidation_priority"], 4),
                "prospective_memory_pull": round(memory_orchestration["prospective_memory_pull"], 4),
                "working_memory_pressure": round(settled_working_memory.memory_pressure, 4),
                "unresolved_count": int(settled_working_memory.unresolved_count),
                "current_focus": settled_working_memory.current_focus,
                "pending_meaning": round(settled_working_memory.pending_meaning, 4),
                "affective_position": terrain_position_state.to_dict(),
                "terrain_readout": terrain_readout_before.to_dict(),
                "protection_mode": dict(protection_mode_payload),
                "memory_write_class": memory_write_class,
                "memory_write_class_reason": memory_write_class_reason,
                "commitment_state": dict(commitment_state_payload),
                "agenda_state": dict(agenda_state_payload),
                "initiative_followup_bias": dict(initiative_followup_bias),
                "temperament_trace": dict(temperament_trace),
                "terrain_plasticity_applied": terrain_plasticity_update is not None,
                "terrain_plasticity_signals": dict(terrain_plasticity_signals),
                "terrain_plasticity_update": (
                    terrain_plasticity_update.to_dict()
                    if terrain_plasticity_update is not None
                    else {}
                ),
                "affective_terrain_state": terrain_state_after.to_dict(),
                "association_graph_state": association_graph_state_after.to_dict(),
                "insight_event": dict(insight_event_payload),
                "insight_trace": insight_trace.to_dict() if insight_trace is not None else {},
                "candidate_continuity_pull": round(candidate_signal["continuity_pull"], 4),
                "candidate_meaning_pull": round(candidate_signal["meaning_pull"], 4),
                "candidate_social_pull": round(candidate_signal["social_pull"], 4),
                "candidate_focus_hint": candidate_signal["focus_hint"],
                "candidate_anchor_hint": candidate_signal["anchor_hint"],
                "candidate_target_person_id": candidate_signal["target_person_id"],
                "person_registry_person_id": candidate_signal["target_person_id"] or counterpart_person_id,
                "interaction_alignment_score": round(interaction_alignment["alignment_score"], 4),
                "shared_attention_delta": round(interaction_alignment["shared_attention_delta"], 4),
                "distance_mismatch": round(interaction_alignment["distance_mismatch"], 4),
                "hesitation_mismatch": round(interaction_alignment["hesitation_mismatch"], 4),
                "opening_pace_mismatch": round(interaction_alignment["opening_pace_mismatch"], 4),
                "return_gaze_mismatch": round(interaction_alignment["return_gaze_mismatch"], 4),
                "observed_gaze_mode": interaction_alignment["observed_gaze_mode"],
                "observed_pause_mode": interaction_alignment["observed_pause_mode"],
                "observed_proximity_mode": interaction_alignment["observed_proximity_mode"],
                "observed_opening_pace": interaction_alignment["observed_opening_pace"],
                "observed_return_gaze": interaction_alignment["observed_return_gaze"],
                "observed_trace_cues": list(interaction_alignment["observed_trace_cues"]),
                "long_term_theme_focus": settled_working_memory.long_term_theme_focus,
                "long_term_theme_anchor": settled_working_memory.long_term_theme_anchor,
                "long_term_theme_strength": round(settled_working_memory.long_term_theme_strength, 4),
                "working_memory_replay_focus": replay_signature_focus,
                "working_memory_replay_anchor": replay_signature_anchor,
                  "working_memory_replay_strength": round(replay_signature_strength, 4),
                  "working_memory_replay_alignment": round(replay_signature_alignment, 4),
                  "working_memory_replay_reinforcement": round(replay_signature_reinforcement, 4),
                  "long_term_theme_summary": settled_working_memory.long_term_theme_summary,
                  "long_term_theme_alignment": round(long_term_theme_alignment, 4),
                  "long_term_theme_reinforcement": round(long_term_theme_reinforcement, 4),
                  "memory_orchestration": dict(memory_orchestration),
                  "environment_pressure": environment_pressure.to_dict(),
              },
            person_registry_snapshot=person_registry_snapshot,
            group_thread_registry_snapshot=group_thread_registry_snapshot,
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


def _default_affective_position_dim(axis_labels: Any) -> int:
    axis_count = len(tuple(axis_labels or ()))
    if axis_count <= 0:
        return 4
    return max(1, min(4, axis_count))


def _coerce_affective_position_state(
    value: Any,
    *,
    default_dim: int,
) -> AffectivePositionState:
    if isinstance(value, AffectivePositionState):
        return value
    if not isinstance(value, Mapping):
        return make_neutral_affective_position(default_dim)
    z_aff = np.asarray(list(value.get("z_aff") or []), dtype=np.float32).reshape(-1)
    if z_aff.size == 0:
        z_aff = np.zeros(default_dim, dtype=np.float32)
    if z_aff.size < default_dim:
        z_aff = np.concatenate([z_aff, np.zeros(default_dim - z_aff.size, dtype=np.float32)])
    elif z_aff.size > default_dim:
        z_aff = z_aff[:default_dim]
    cov_payload = value.get("cov") or []
    cov = np.asarray(cov_payload, dtype=np.float32)
    if cov.shape != (z_aff.size, z_aff.size):
        cov = np.eye(z_aff.size, dtype=np.float32)
    confidence = _clamp01(_float_from(value, "confidence", default=0.0))
    source_weights_payload = value.get("source_weights") or {}
    source_weights = (
        {
            str(key): _clamp01(float(item))
            for key, item in source_weights_payload.items()
            if str(key).strip()
        }
        if isinstance(source_weights_payload, Mapping)
        else {"carryover": 1.0}
    )
    if not source_weights:
        source_weights = {"carryover": 1.0}
    return AffectivePositionState(
        z_aff=z_aff.astype(np.float32),
        cov=cov.astype(np.float32),
        confidence=confidence,
        source_weights=source_weights,
    )


def _coerce_affective_terrain_state(
    value: Any,
    *,
    position_dim: int,
) -> AffectiveTerrainState:
    if isinstance(value, AffectiveTerrainState):
        return value
    if not isinstance(value, Mapping):
        return make_neutral_affective_terrain_state(position_dim=position_dim)
    try:
        terrain = AffectiveTerrainState(
            centers=np.asarray(value.get("centers") or [], dtype=np.float32),
            widths=np.asarray(value.get("widths") or [], dtype=np.float32).reshape(-1),
            value_weights=np.asarray(value.get("value_weights") or [], dtype=np.float32).reshape(-1),
            approach_weights=np.asarray(value.get("approach_weights") or [], dtype=np.float32).reshape(-1),
            avoid_weights=np.asarray(value.get("avoid_weights") or [], dtype=np.float32).reshape(-1),
            protect_weights=np.asarray(value.get("protect_weights") or [], dtype=np.float32).reshape(-1),
            anchor_labels=tuple(str(item) for item in value.get("anchor_labels") or [] if str(item).strip()),
        )
    except Exception:
        return make_neutral_affective_terrain_state(position_dim=position_dim)
    if terrain.position_dim != position_dim:
        return make_neutral_affective_terrain_state(position_dim=position_dim, patch_count=terrain.patch_count)
    return terrain


def _build_affective_memory_echo(current_state: Mapping[str, Any]) -> list[float]:
    return [
        _float_from(current_state, "interaction_afterglow", default=0.0),
        _float_from(current_state, "replay_intensity", default=0.0),
        _float_from(current_state, "meaning_inertia", default=0.0),
        _float_from(current_state, "conscious_residue_strength", default=0.0),
    ]


def _has_affective_terrain_context(current_state: Mapping[str, Any]) -> bool:
    position = current_state.get("prev_affective_position") or current_state.get("affective_position")
    terrain = current_state.get("affective_terrain_state")
    return isinstance(position, (AffectivePositionState, Mapping)) and isinstance(
        terrain, (AffectiveTerrainState, Mapping)
    )


def _coerce_terrain_readout(
    value: Any,
    *,
    terrain_state: AffectiveTerrainState,
    position_state: AffectivePositionState,
) -> TerrainReadout:
    if isinstance(value, TerrainReadout):
        return value
    if isinstance(value, Mapping):
        grad = np.asarray(list(value.get("grad") or []), dtype=np.float32).reshape(-1)
        curvature = np.asarray(list(value.get("curvature") or []), dtype=np.float32).reshape(-1)
        position_dim = terrain_state.position_dim
        if grad.size < position_dim:
            grad = np.concatenate([grad, np.zeros(position_dim - grad.size, dtype=np.float32)])
        elif grad.size > position_dim:
            grad = grad[:position_dim]
        if curvature.size < position_dim:
            curvature = np.concatenate(
                [curvature, np.zeros(position_dim - curvature.size, dtype=np.float32)]
            )
        elif curvature.size > position_dim:
            curvature = curvature[:position_dim]
        return TerrainReadout(
            value=float(value.get("value", 0.0) or 0.0),
            grad=grad.astype(np.float32),
            curvature=curvature.astype(np.float32),
            approach_bias=_clamp01(_float_from(value, "approach_bias", default=0.0)),
            avoid_bias=_clamp01(_float_from(value, "avoid_bias", default=0.0)),
            protect_bias=_clamp01(_float_from(value, "protect_bias", default=0.0)),
            active_patch_index=int(value.get("active_patch_index", 0) or 0),
            active_patch_label=str(value.get("active_patch_label") or ""),
        )
    return BasicAffectiveTerrain().read(terrain_state, position_state)


def _derive_terrain_plasticity_signals(
    *,
    next_state: HookState,
    interaction_alignment: Mapping[str, Any],
    candidate_signal: Mapping[str, Any],
    terrain_readout: TerrainReadout,
    protection_mode: Mapping[str, Any],
    reply_present: bool,
) -> Dict[str, float]:
    protection_mode_name = str(protection_mode.get("mode") or "").strip()
    protection_mode_strength = _float_from(protection_mode, "strength", default=0.0)
    unresolved_scale = _clamp01(float(next_state.unresolved_count) / 3.0)

    safety_gain = _clamp01(
        float(interaction_alignment.get("alignment_score") or 0.0) * 0.42
        + next_state.continuity_score * 0.16
        + next_state.social_grounding * 0.1
        + _float_from(candidate_signal, "continuity_pull", default=0.0) * 0.12
        + _float_from(candidate_signal, "social_pull", default=0.0) * 0.08
        + (0.08 if reply_present else 0.0)
        + (0.08 if protection_mode_name in {"repair", "monitor"} else 0.0)
        - next_state.recent_strain * 0.1
    )
    strain_load = _clamp01(
        next_state.stress * 0.26
        + next_state.recovery_need * 0.16
        + next_state.recent_strain * 0.22
        + float(interaction_alignment.get("mismatch_intensity") or 0.0) * 0.16
        + terrain_readout.protect_bias * 0.12
        + protection_mode_strength
        * (0.08 if protection_mode_name in {"contain", "stabilize", "shield"} else 0.0)
    )
    bond_weight = _clamp01(
        next_state.attachment * 0.42
        + next_state.trust_memory * 0.24
        + next_state.familiarity * 0.12
        + _float_from(candidate_signal, "social_pull", default=0.0) * 0.14
        + (0.08 if next_state.related_person_id else 0.0)
    )
    unresolved_tension = _clamp01(
        next_state.pending_meaning * 0.4
        + unresolved_scale * 0.2
        + next_state.meaning_inertia * 0.18
        + float(interaction_alignment.get("mismatch_intensity") or 0.0) * 0.12
        + (0.08 if protection_mode_name in {"contain", "stabilize", "shield"} else 0.0)
    )
    return {
        "safety_gain": round(safety_gain, 4),
        "strain_load": round(strain_load, 4),
        "bond_weight": round(bond_weight, 4),
        "unresolved_tension": round(unresolved_tension, 4),
    }


def _replay_signature_alignment(
    *,
    recall_payload: Mapping[str, Any],
    current_state: Mapping[str, Any],
    focus: Optional[str],
    anchor: Optional[str],
) -> float:
    focus_text = str(focus or "").strip().lower()
    anchor_text = str(anchor or "").strip().lower()
    haystack = " ".join(
        str((recall_payload or {}).get(key) or "")
        for key in ("summary", "text", "memory_anchor", "reinterpretation_summary")
    ).lower()
    recall_anchor = str((recall_payload or {}).get("memory_anchor") or (current_state or {}).get("memory_anchor") or "").strip().lower()
    alignment = 0.0
    if focus_text and focus_text in haystack:
        alignment += 0.55
    if anchor_text and anchor_text == recall_anchor:
        alignment += 0.45
    return _clamp01(alignment)


def _expression_hints(*, terrain_transition_roughness: float, caution_bias: float, recent_strain: float, continuity_score: float, social_update_strength: float = 1.0, identity_update_strength: float = 1.0, interaction_afterglow: float = 0.0, interaction_afterglow_intent: Optional[str] = None, replay_intensity: float = 0.0, anticipation_tension: float = 0.0, stabilization_drive: float = 0.0, relational_clarity: float = 0.0, meaning_inertia: float = 0.0, recovery_reopening: float = 0.0, object_affordance_bias: float = 0.0, fragility_guard: float = 0.0, object_attachment: float = 0.0, object_avoidance: float = 0.0, tool_extension_bias: float = 0.0, ritually_sensitive_bias: float = 0.0, defensive_salience: float = 0.0, reachability: float = 0.0, long_term_theme_strength: float = 0.0, long_term_theme_kind: str = "") -> Dict[str, Any]:
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
        - long_term_theme_strength * 0.06
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
        "long_term_theme_strength": round(long_term_theme_strength, 4),
        "long_term_theme_kind": long_term_theme_kind,
    }


def _terms(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in str(text or "").lower().replace("\n", " ").split():
        token = raw.strip(".,!?;:'\"()[]{}")
        if len(token) >= 3 and token not in tokens:
            tokens.append(token)
    return tokens


def _long_term_theme_alignment(
    *,
    recall_payload: Mapping[str, Any],
    reply_text: str,
    summary: Optional[str],
) -> float:
    target = str(summary or "").strip()
    if not target:
        return 0.0
    tokens = _terms(target)
    if not tokens:
        return 0.0
    haystack = " ".join(
        filter(
            None,
            [
                str(recall_payload.get("summary") or "").strip(),
                str(recall_payload.get("text") or "").strip(),
                str(recall_payload.get("memory_anchor") or "").strip(),
                str(reply_text or "").strip(),
            ],
        )
    ).lower()
    if not haystack:
        return 0.0
    hits = sum(1 for token in tokens[:4] if token and token in haystack)
    return _clamp01(hits / max(1, min(len(tokens), 4)))

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


def _memory_candidate_influence(candidates: list[Mapping[str, Any]]) -> Dict[str, Any]:
    continuity_pull = 0.0
    meaning_pull = 0.0
    social_pull = 0.0
    focus_hint = ""
    anchor_hint = ""
    target_person_id = ""
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        kind = str(item.get("kind") or "").strip().lower()
        summary = str(item.get("summary") or item.get("text") or "").strip()
        anchor = str(item.get("memory_anchor") or "").strip()
        related_person_id = str(item.get("related_person_id") or "").strip()
        policy_hint = str(item.get("policy_hint") or "").strip().lower()
        continuity_score = _float_from(item, "continuity_score", default=0.0)
        consolidation_priority = _float_from(item, "consolidation_priority", default=0.0)
        meaning_shift = _float_from(item, "meaning_shift", default=0.0)
        if continuity_score > 0.0 or "continuity" in policy_hint:
            continuity_pull += max(continuity_score, 0.45)
            if not focus_hint:
                focus_hint = "meaning"
        if any(tag in policy_hint for tag in ("social", "relation", "affiliation", "trust")):
            social_pull += 0.42
            if not focus_hint:
                focus_hint = "social"
        if related_person_id and not target_person_id:
            target_person_id = related_person_id
        if kind == "reconstructed" or meaning_shift > 0.0 or "semantic_hint" in policy_hint:
            meaning_pull += max(meaning_shift, consolidation_priority, 0.36)
            if not focus_hint:
                focus_hint = "meaning"
        elif consolidation_priority > 0.0:
            meaning_pull += consolidation_priority * 0.6
        if not anchor_hint and anchor:
            anchor_hint = anchor
        if not anchor_hint and summary:
            anchor_hint = summary[:160]
    return {
        "continuity_pull": round(_clamp01(continuity_pull * 0.38), 4),
        "meaning_pull": round(_clamp01(meaning_pull * 0.34), 4),
        "social_pull": round(_clamp01(social_pull * 0.32), 4),
        "focus_hint": focus_hint,
        "anchor_hint": anchor_hint[:160],
        "target_person_id": target_person_id[:120],
    }


def _interaction_alignment_snapshot(
    *,
    current_state: Mapping[str, Any],
    output: Mapping[str, Any],
) -> Dict[str, Any]:
    predicted_nonverbal = _mapping_or_empty(current_state.get("predicted_nonverbal"))
    observed_trace = summarize_interaction_trace(
        sensor_input=output,
        current_state=current_state,
    )
    predicted_gaze_mode = _text_or_none(predicted_nonverbal.get("gaze_mode")) or _text_or_none(current_state.get("predicted_gaze_mode")) or ""
    predicted_pause_mode = _text_or_none(predicted_nonverbal.get("pause_mode")) or _text_or_none(current_state.get("predicted_pause_mode")) or ""
    predicted_proximity_mode = _text_or_none(predicted_nonverbal.get("proximity_mode")) or _text_or_none(current_state.get("predicted_distance_expectation")) or ""
    predicted_shared_attention = _float_from(
        current_state,
        "predicted_shared_attention",
        default=_float_from(predicted_nonverbal, "shared_attention", default=0.0),
    )
    predicted_hesitation = _text_or_none(current_state.get("predicted_hesitation_tone")) or predicted_pause_mode
    predicted_opening_pace = _text_or_none(current_state.get("opening_pace_windowed")) or ""
    predicted_return_gaze = _text_or_none(current_state.get("return_gaze_expectation")) or ""

    observed_gaze_mode = observed_trace.gaze_mode
    observed_pause_mode = observed_trace.pause_mode
    observed_proximity_mode = observed_trace.proximity_mode
    observed_shared_attention = observed_trace.shared_attention if observed_trace.shared_attention > 0.0 else _float_from(
        current_state,
        "observed_shared_attention",
        default=predicted_shared_attention,
    )
    observed_hesitation = observed_trace.hesitation_tone or observed_pause_mode
    observed_opening_pace = _opening_pace_windowed(
        strained_pause_window_mean=_observed_strained_pause_window_mean(output=output, observed_trace=observed_trace),
        repair_window_hold=_observed_repair_window_hold(output=output, observed_trace=observed_trace),
    )
    observed_return_gaze = _return_gaze_expectation(
        shared_attention_window_mean=_observed_shared_attention_window_mean(output=output, observed_trace=observed_trace),
        repair_window_hold=_observed_repair_window_hold(output=output, observed_trace=observed_trace),
        relational_reverence=_float_from(current_state, "relational_reverence", default=0.0),
    )

    gaze_alignment = _categorical_alignment(
        predicted_gaze_mode,
        observed_gaze_mode,
        similar_groups=(
            {"shared_attention_hold", "soft_hold", "respectful_glance"},
            {"avert", "guarded_glance"},
        ),
    )
    pause_alignment = _categorical_alignment(
        predicted_pause_mode,
        observed_pause_mode,
        similar_groups=(
            {"short_warm", "confident_brief", "future_opening"},
            {"patient_care", "tender_tentative", "measured_ritual"},
        ),
    )
    proximity_alignment = _categorical_alignment(
        predicted_proximity_mode,
        observed_proximity_mode,
        similar_groups=(
            {"gentle_near", "shared_world_orientation"},
            {"measured_distance", "holding_space"},
        ),
    )
    hesitation_alignment = _categorical_alignment(
        predicted_hesitation,
        observed_hesitation,
        similar_groups=(
            {"short_warm", "confident_brief"},
            {"patient_care", "measured_ritual", "tender_tentative"},
        ),
    )
    shared_attention_alignment = 1.0 - abs(_clamp01(observed_shared_attention) - _clamp01(predicted_shared_attention))
    opening_pace_alignment = _categorical_alignment(
        predicted_opening_pace,
        observed_opening_pace,
        similar_groups=(
            {"held", "measured"},
            {"ready", "measured"},
        ),
    )
    return_gaze_alignment = _categorical_alignment(
        predicted_return_gaze,
        observed_return_gaze,
        similar_groups=(
            {"soft_return", "steady_return"},
            {"careful_return", "defer_return"},
        ),
    )
    alignment_score = _clamp01(
        gaze_alignment * 0.2
        + pause_alignment * 0.18
        + proximity_alignment * 0.14
        + hesitation_alignment * 0.14
        + shared_attention_alignment * 0.14
        + opening_pace_alignment * 0.1
        + return_gaze_alignment * 0.1
    )
    distance_mismatch = round(1.0 - proximity_alignment, 4)
    hesitation_mismatch = round(1.0 - hesitation_alignment, 4)
    shared_attention_delta = round(abs(_clamp01(observed_shared_attention) - _clamp01(predicted_shared_attention)), 4)
    opening_pace_mismatch = round(1.0 - opening_pace_alignment, 4)
    return_gaze_mismatch = round(1.0 - return_gaze_alignment, 4)
    mismatch_intensity = _clamp01(
        distance_mismatch * 0.24
        + hesitation_mismatch * 0.22
        + (1.0 - gaze_alignment) * 0.14
        + shared_attention_delta * 0.16
        + opening_pace_mismatch * 0.12
        + return_gaze_mismatch * 0.12
    )
    return {
        "alignment_score": round(alignment_score, 4),
        "mismatch_intensity": round(mismatch_intensity, 4),
        "shared_attention_delta": shared_attention_delta,
        "distance_mismatch": distance_mismatch,
        "hesitation_mismatch": hesitation_mismatch,
        "opening_pace_mismatch": opening_pace_mismatch,
        "return_gaze_mismatch": return_gaze_mismatch,
        "observed_gaze_mode": observed_gaze_mode,
        "observed_pause_mode": observed_pause_mode,
        "observed_proximity_mode": observed_proximity_mode,
        "observed_opening_pace": observed_opening_pace,
        "observed_return_gaze": observed_return_gaze,
        "observed_trace_cues": list(observed_trace.cues),
    }


def _categorical_alignment(
    predicted: Optional[str],
    observed: Optional[str],
    *,
    similar_groups: tuple[set[str], ...] = (),
) -> float:
    lhs = str(predicted or "").strip().lower()
    rhs = str(observed or "").strip().lower()
    if not lhs or not rhs:
        return 0.5
    if lhs == rhs:
        return 1.0
    for group in similar_groups:
        if lhs in group and rhs in group:
            return 0.68
    return 0.0


def _person_registry_person_state(
    registry_snapshot: Mapping[str, Any],
    person_id: Optional[str],
) -> Dict[str, Any]:
    target_id = _text_or_none(person_id)
    if not target_id:
        return {}
    persons = registry_snapshot.get("persons") if isinstance(registry_snapshot, Mapping) else None
    if not isinstance(persons, Mapping):
        return {}
    node = persons.get(target_id)
    if not isinstance(node, Mapping):
        return {}
    adaptive_traits = node.get("adaptive_traits")
    if not isinstance(adaptive_traits, Mapping):
        return {}
    state: Dict[str, Any] = {}
    for key in (
        "attachment",
        "familiarity",
        "trust_memory",
        "continuity_score",
        "social_grounding",
        "style_warmth_memory",
        "playful_ceiling",
        "advice_tolerance",
        "lexical_familiarity",
        "lexical_variation_bias",
    ):
        if key in adaptive_traits:
            state[key] = adaptive_traits.get(key)
    return state


def _derive_registry_style_traits(
    *,
    current_state: Mapping[str, Any],
    next_state: HookState,
    reply_present: bool,
) -> Dict[str, float]:
    partner_address_hint = _text_or_none(current_state.get("partner_address_hint")) or ""
    partner_timing_hint = _text_or_none(current_state.get("partner_timing_hint")) or ""
    partner_stance_hint = _text_or_none(current_state.get("partner_stance_hint")) or ""
    partner_social_interpretation = _text_or_none(current_state.get("partner_social_interpretation")) or ""

    familiar_open = 0.0
    careful_distance = 0.0
    if partner_stance_hint == "familiar":
        familiar_open += 0.18
    elif partner_stance_hint == "respectful":
        careful_distance += 0.16
    if partner_timing_hint == "open":
        familiar_open += 0.14
    elif partner_timing_hint == "delayed":
        careful_distance += 0.18
    if partner_address_hint == "companion":
        familiar_open += 0.12
    elif partner_address_hint in {"senpai", "respectful"}:
        careful_distance += 0.12
    if "familiar" in partner_social_interpretation or "future_open" in partner_social_interpretation:
        familiar_open += 0.1
    if "delayed" in partner_social_interpretation or "respectful" in partner_social_interpretation:
        careful_distance += 0.1

    relation_bias = _clamp01(
        next_state.attachment * 0.34
        + next_state.familiarity * 0.22
        + next_state.trust_memory * 0.22
        + next_state.continuity_score * 0.12
        + next_state.social_grounding * 0.1
    )
    style_warmth_memory = _clamp01(
        0.18
        + next_state.attachment * 0.22
        + next_state.trust_memory * 0.18
        + relation_bias * 0.12
        + familiar_open * 0.16
        - careful_distance * 0.08
    )
    playful_ceiling = _clamp01(
        0.14
        + next_state.familiarity * 0.2
        + next_state.exploration_bias * 0.16
        + relation_bias * 0.08
        + familiar_open * 0.16
        - careful_distance * 0.16
        - next_state.recent_strain * 0.08
        + (0.05 if reply_present else 0.0)
    )
    advice_tolerance = _clamp01(
        0.16
        + next_state.trust_memory * 0.18
        + next_state.continuity_score * 0.16
        + next_state.social_grounding * 0.08
        + familiar_open * 0.1
        - careful_distance * 0.12
        - next_state.recent_strain * 0.08
        - (0.04 if not reply_present else 0.0)
    )
    lexical_familiarity = _clamp01(
        0.14
        + next_state.familiarity * 0.22
        + next_state.continuity_score * 0.18
        + relation_bias * 0.1
        + familiar_open * 0.08
        - careful_distance * 0.06
        + (0.06 if reply_present else 0.0)
    )
    lexical_variation_bias = _clamp01(
        lexical_familiarity * 0.48
        + playful_ceiling * 0.14
        + familiar_open * 0.08
        - careful_distance * 0.12
        - next_state.recent_strain * 0.1
    )
    return {
        "style_warmth_memory": round(style_warmth_memory, 4),
        "playful_ceiling": round(playful_ceiling, 4),
        "advice_tolerance": round(advice_tolerance, 4),
        "lexical_familiarity": round(lexical_familiarity, 4),
        "lexical_variation_bias": round(lexical_variation_bias, 4),
    }


def _updated_person_registry_snapshot(
    *,
    existing_snapshot: Mapping[str, Any],
    target_person_id: Optional[str],
    attachment: float,
    familiarity: float,
    trust_memory: float,
    continuity_score: float,
    social_grounding: float,
    current_focus: str,
    community_id: str,
    culture_id: str,
    social_role: str,
    style_warmth_memory: float,
    playful_ceiling: float,
    advice_tolerance: float,
    lexical_familiarity: float,
    lexical_variation_bias: float,
) -> Dict[str, Any]:
    person_id = _text_or_none(target_person_id)
    if not person_id:
        return dict(existing_snapshot or {})
    existing_persons = dict(existing_snapshot.get("persons") or {}) if isinstance(existing_snapshot, Mapping) else {}
    registry = PersonRegistry(
        persons={
            str(key): _registry_node_from_snapshot(str(key), value)
            for key, value in existing_persons.items()
            if isinstance(value, Mapping)
        },
        uncertainty=float(existing_snapshot.get("uncertainty", 1.0) or 1.0) if isinstance(existing_snapshot, Mapping) else 1.0,
    )
    updated = update_person_registry(
        registry,
        {
            "summary": f"{person_id}:{current_focus or 'ongoing'}",
            "adaptive_traits": {
                "attachment": round(attachment, 4),
                "familiarity": round(familiarity, 4),
                "trust_memory": round(trust_memory, 4),
                "continuity_score": round(continuity_score, 4),
                "social_grounding": round(social_grounding, 4),
                "style_warmth_memory": round(style_warmth_memory, 4),
                "playful_ceiling": round(playful_ceiling, 4),
                "advice_tolerance": round(advice_tolerance, 4),
                "lexical_familiarity": round(lexical_familiarity, 4),
                "lexical_variation_bias": round(lexical_variation_bias, 4),
            },
            "stable_traits": {
                "community_marker": 1.0 if community_id else 0.0,
                "culture_marker": 1.0 if culture_id else 0.0,
                "role_marker": 1.0 if social_role else 0.0,
            },
            "ambiguity": 0.18,
        },
        {"person_id": person_id},
    )
    serialized_persons: Dict[str, Any] = {}
    for key, node in updated.persons.items():
        serialized_persons[key] = {
            "person_id": node.person_id,
            "stable_traits": dict(node.stable_traits),
            "adaptive_traits": dict(node.adaptive_traits),
            "continuity_history": list(node.continuity_history),
            "confidence": node.confidence,
            "ambiguity_flag": node.ambiguity_flag,
        }
    return {
        "persons": serialized_persons,
        "uncertainty": updated.uncertainty,
        **summarize_person_registry_snapshot(
            {
                "persons": serialized_persons,
                "uncertainty": updated.uncertainty,
            }
        ),
    }


def _updated_group_thread_registry_snapshot(
    *,
    existing_snapshot: Mapping[str, Any],
    thread_hint: str,
    topology_state: Mapping[str, Any],
    dominant_person_id: str,
    top_person_ids: list[str],
    total_people: int,
    continuity_score: float,
    social_grounding: float,
    community_id: str,
    culture_id: str,
    social_role: str,
) -> Dict[str, Any]:
    return advance_group_thread_registry_snapshot(
        existing_snapshot=existing_snapshot,
        thread_hint=thread_hint,
        topology_state=topology_state,
        dominant_person_id=dominant_person_id,
        top_person_ids=top_person_ids,
        total_people=total_people,
        continuity_score=continuity_score,
        social_grounding=social_grounding,
        community_id=community_id,
        culture_id=culture_id,
        social_role=social_role,
    )


def _registry_node_from_snapshot(person_id: str, payload: Mapping[str, Any]) -> PersonNode:
    return PersonNode(
        person_id=person_id,
        stable_traits=dict(payload.get("stable_traits") or {}),
        adaptive_traits=dict(payload.get("adaptive_traits") or {}),
        continuity_history=list(payload.get("continuity_history") or []),
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        ambiguity_flag=bool(payload.get("ambiguity_flag", True)),
    )


def _opening_pace_windowed(
    *,
    strained_pause_window_mean: float,
    repair_window_hold: float,
) -> str:
    if repair_window_hold >= 0.44 or strained_pause_window_mean >= 0.52:
        return "held"
    if repair_window_hold >= 0.24 or strained_pause_window_mean >= 0.32:
        return "measured"
    return "ready"


def _return_gaze_expectation(
    *,
    shared_attention_window_mean: float,
    repair_window_hold: float,
    relational_reverence: float,
) -> str:
    if repair_window_hold >= 0.4:
        return "defer_return"
    if relational_reverence >= 0.56:
        return "careful_return"
    if shared_attention_window_mean >= 0.54:
        return "steady_return"
    return "soft_return"


def _observed_shared_attention_window_mean(
    *,
    output: Mapping[str, Any],
    observed_trace: Any,
) -> float:
    direct = _float_from(output, "observed_shared_attention_window_mean", default=-1.0)
    if direct >= 0.0:
        return _clamp01(direct)
    return _clamp01(observed_trace.shared_attention)


def _observed_strained_pause_window_mean(
    *,
    output: Mapping[str, Any],
    observed_trace: Any,
) -> float:
    direct = _float_from(output, "observed_strained_pause_window_mean", default=-1.0)
    if direct >= 0.0:
        return _clamp01(direct)
    pause_latency = _float_from(output, "pause_latency", default=0.0)
    hesitation_signal = _float_from(output, "hesitation_signal", default=0.0)
    if pause_latency > 0.0 or hesitation_signal > 0.0:
        return _clamp01(pause_latency * 0.65 + hesitation_signal * 0.35)
    if observed_trace.pause_mode in {"measured_ritual", "waiting"}:
        return 0.38
    if observed_trace.pause_mode in {"patient_care", "tender_tentative"}:
        return 0.28
    return 0.14


def _observed_repair_window_hold(
    *,
    output: Mapping[str, Any],
    observed_trace: Any,
) -> float:
    direct = _float_from(output, "observed_repair_window_hold", default=-1.0)
    if direct >= 0.0:
        return _clamp01(direct)
    repair_signal = _float_from(output, "repair_signal", default=0.0)
    if repair_signal > 0.0:
        return _clamp01(repair_signal)
    if "repair_signal_detected" in list(observed_trace.cues):
        return 0.42
    return 0.0


def _derive_initiative_followup_bias(
    *,
    body_recovery_guard: Mapping[str, Any],
    initiative_readiness: Mapping[str, Any],
    commitment_state: Mapping[str, Any],
    protection_mode: Mapping[str, Any],
    temperament_estimate: Mapping[str, Any],
    memory_write_class: str,
    pending_meaning: float,
    prospective_memory_pull: float,
    reply_present: bool,
) -> Dict[str, Any]:
    guard_state = str(body_recovery_guard.get("state") or "open").strip() or "open"
    guard_score = _clamp01(float(body_recovery_guard.get("score", 0.0) or 0.0))
    readiness_state = str(initiative_readiness.get("state") or "hold").strip() or "hold"
    readiness_score = _clamp01(float(initiative_readiness.get("score", 0.0) or 0.0))
    commitment_mode = str(commitment_state.get("state") or "waver").strip() or "waver"
    commitment_target = str(commitment_state.get("target") or "hold").strip() or "hold"
    commitment_score = _clamp01(float(commitment_state.get("score", 0.0) or 0.0))
    mode = str(protection_mode.get("mode") or "").strip()
    mode_strength = _clamp01(float(protection_mode.get("strength", 0.0) or 0.0))
    temperament = dict(temperament_estimate or {})
    risk_tolerance = _clamp01(float(temperament.get("risk_tolerance", 0.0) or 0.0))
    bond_drive = _clamp01(float(temperament.get("bond_drive", 0.0) or 0.0))
    curiosity_drive = _clamp01(float(temperament.get("curiosity_drive", 0.0) or 0.0))
    recovery_discipline = _clamp01(float(temperament.get("recovery_discipline", 0.0) or 0.0))
    protect_floor = _clamp01(float(temperament.get("protect_floor", 0.0) or 0.0))
    initiative_persistence = _clamp01(float(temperament.get("initiative_persistence", 0.0) or 0.0))
    pending_meaning = _clamp01(pending_meaning)
    prospective_memory_pull = _clamp01(prospective_memory_pull)
    memory_write_class = str(memory_write_class or "episodic").strip() or "episodic"
    reply_signal = 1.0 if reply_present else 0.0
    commitment_commit = 1.0 if commitment_mode == "commit" else 0.0
    commitment_settle = 1.0 if commitment_mode == "settle" else 0.0

    hold_score = _clamp01(
        0.3 * guard_score
        + 0.22 * mode_strength * (1.0 if mode in {"shield", "stabilize"} else 0.0)
        + 0.16 * (1.0 if guard_state == "recovery_first" else 0.0)
        + 0.08 * protect_floor
        + 0.08 * recovery_discipline * guard_score
        + 0.12 * commitment_score * commitment_commit * (1.0 if commitment_target in {"hold", "stabilize"} else 0.0)
        + 0.06 * commitment_score * commitment_settle * (1.0 if commitment_target == "hold" else 0.0)
    )
    reopen_softly_score = _clamp01(
        0.18 * readiness_score * (1.0 if readiness_state in {"tentative", "ready"} else 0.0)
        + 0.2 * pending_meaning
        + 0.16 * prospective_memory_pull
        + 0.1 * reply_signal
        + 0.1 * (1.0 if memory_write_class in {"repair_trace", "bond_protection", "safe_repeat"} else 0.0)
        + 0.08 * bond_drive
        + 0.12 * commitment_score * commitment_commit * (1.0 if commitment_target in {"repair", "bond_protect"} else 0.0)
        + 0.06 * commitment_score * commitment_settle * (1.0 if commitment_target in {"repair", "bond_protect"} else 0.0)
        - 0.12 * guard_score
        - 0.08 * mode_strength * (1.0 if mode == "shield" else 0.0)
    )
    offer_next_step_score = _clamp01(
        0.22 * readiness_score * (1.0 if readiness_state == "ready" else 0.0)
        + 0.16 * prospective_memory_pull
        + 0.12 * pending_meaning
        + 0.08 * reply_signal
        + 0.08 * (1.0 if memory_write_class in {"safe_repeat", "repair_trace", "insight_trace"} else 0.0)
        + 0.1 * risk_tolerance
        + 0.08 * curiosity_drive
        + 0.08 * initiative_persistence
        + 0.14 * commitment_score * commitment_commit * (1.0 if commitment_target == "step_forward" else 0.0)
        + 0.06 * commitment_score * commitment_settle * (1.0 if commitment_target == "step_forward" else 0.0)
        - 0.18 * guard_score
        - 0.12 * mode_strength * (1.0 if mode in {"contain", "stabilize", "shield"} else 0.0)
        - 0.1 * protect_floor
        - 0.08 * recovery_discipline * guard_score
    )
    if guard_state == "recovery_first":
        reopen_softly_score = _clamp01(reopen_softly_score * 0.72)
        offer_next_step_score = _clamp01(offer_next_step_score * 0.25)
    elif guard_state == "guarded":
        offer_next_step_score = _clamp01(offer_next_step_score * 0.55)
    if commitment_target in {"hold", "stabilize"} and commitment_commit >= 1.0:
        reopen_softly_score = _clamp01(reopen_softly_score * 0.82)
        offer_next_step_score = _clamp01(offer_next_step_score * 0.6)

    scores = {
        "hold": hold_score,
        "reopen_softly": reopen_softly_score,
        "offer_next_step": offer_next_step_score,
    }
    state, winner_margin = _winner_and_margin(scores)
    dominant_inputs = [
        item
        for item in (
            "body_recovery_guard" if guard_state in {"guarded", "recovery_first"} else "",
            "initiative_readiness" if readiness_state in {"tentative", "ready"} and readiness_score >= 0.2 else "",
            "commitment_step_forward" if state == "offer_next_step" and commitment_target == "step_forward" and commitment_score >= 0.2 else "",
            "commitment_repair" if state == "reopen_softly" and commitment_target in {"repair", "bond_protect"} and commitment_score >= 0.2 else "",
            "commitment_hold" if state == "hold" and commitment_target in {"hold", "stabilize"} and commitment_score >= 0.2 else "",
            "temperament_forward_trace" if state == "offer_next_step" and risk_tolerance >= 0.62 else "",
            "temperament_bond_trace" if state == "reopen_softly" and bond_drive >= 0.56 else "",
            "temperament_guard_floor" if state == "hold" and protect_floor >= 0.56 else "",
            "pending_meaning" if pending_meaning >= 0.2 else "",
            "prospective_memory_pull" if prospective_memory_pull >= 0.2 else "",
            "repair_memory" if memory_write_class in {"repair_trace", "bond_protection"} else "",
            "safe_repeat" if memory_write_class == "safe_repeat" else "",
            "protective_mode" if mode in {"contain", "stabilize", "shield"} and mode_strength >= 0.2 else "",
        )
        if item
    ]
    return {
        "state": state,
        "score": round(scores[state], 4),
        "scores": {key: round(value, 4) for key, value in scores.items()},
        "winner_margin": round(winner_margin, 4),
        "dominant_inputs": dominant_inputs,
    }


def _winner_and_margin(scores: Mapping[str, float]) -> tuple[str, float]:
    items = sorted(
        ((str(key), _clamp01(float(value or 0.0))) for key, value in scores.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    if not items:
        return "hold", 0.0
    winner_key, winner_score = items[0]
    runner_up_score = items[1][1] if len(items) > 1 else 0.0
    return winner_key, _clamp01(winner_score - runner_up_score)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
