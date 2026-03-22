"""Reusable inner-life operating system primitives."""

from .conscious_access import ConsciousAccessCore, ConsciousAccessSnapshot
from .hook_contracts import (  # noqa: F401
    MemoryRecallInput,
    PostTurnUpdateInput,
    PreTurnUpdateInput,
    ResponseGateInput,
)
from .integration_hooks import (  # noqa: F401
    HookState,
    IntegrationHooks,
    MemoryRecallResult,
    PostTurnUpdateResult,
    PreTurnUpdateResult,
    ResponseGateResult,
)
from .physiology import (
    BoundaryCore,
    HeartbeatConfig,
    HeartbeatCore,
    HeartbeatState,
    PainStressCore,
    RecoveryCore,
)
from .relational_world import RelationalWorldCore, RelationalWorldState
from .temporal import TemporalWeightCore, TemporalWeightState

from .memory_core import MemoryCore, MemorySearchHit
from .development_core import DevelopmentCore, DevelopmentState
from .reinterpretation_core import ReinterpretationCore, ReinterpretationSnapshot
from .environment_pressure_core import EnvironmentPressureCore, EnvironmentPressureSnapshot
from .relationship_core import RelationshipCore, RelationshipState
from .personality_core import PersonalityIndexCore, PersonalityIndexState
from .persistence_core import PersistenceCore, PersistenceState

from .terrain_core import AffectiveTerrainCore, TerrainSnapshot
from .field_estimator_core import FieldEstimatorCore, FieldEstimateSnapshot
from .working_memory_core import WorkingMemoryCore, WorkingMemorySnapshot

from .simulation_transfer import SimulationEpisode, SimulationTransferCore, TransferredLearning
from .sleep_consolidation_core import SleepConsolidationCore, SleepConsolidationSnapshot
from .service import InnerOSService
from .http_contract import build_inner_os_manifest
from .http_router import router as http_router
from .http_app import app as http_app
from .memory_records import (
    BaseMemoryRecord,
    ObservedRealRecord,
    ReconstructedRecord,
    VerifiedRecord,
    ExperiencedSimRecord,
    TransferredLearningRecord,
    WorkingMemoryTraceRecord,
    normalize_memory_record,
)
from .memory_bridge import collect_runtime_memory_candidates, memory_reference_to_record, observed_vision_to_record
from .kernel_runtime import KERNEL_UPDATE_ORDER, KernelStepContract
from .observation_model import (
    Observation,
    ObservationChannelLayout,
    ObservationChannelSpec,
    ObservationLayout,
    ObservationModel,
    TensorObservationModel,
)
from .action_posture import derive_action_posture
from .actuation_plan import derive_actuation_plan
from .affective_localizer import AffectiveLocalizer, BasicAffectiveLocalizer
from .affective_position import AffectivePositionState, make_neutral_affective_position
from .affective_terrain import (
    AffectiveTerrain,
    AffectiveTerrainState,
    BasicAffectiveTerrain,
    TerrainReadout,
    make_neutral_affective_terrain_state,
)
from .access_dynamics import AccessDynamicRegion, AccessDynamicsState, advance_access_dynamics
from .access_projection import AccessProjection, AccessRegion, project_access_regions
from .affect_blend import AffectBlendState, derive_affect_blend_state
from .agenda_state import AgendaState, derive_agenda_state
from .agenda_window_state import AgendaWindowState, derive_agenda_window_state
from .constraint_field import ConstraintField, derive_constraint_field
from .commitment_state import CommitmentState, derive_commitment_state
from .conscious_workspace import ConsciousWorkspace, WorkspaceSlot, ignite_conscious_workspace
from .conversation_contract import build_conversation_contract
from .conversational_objects import (
    ConversationalObject,
    ConversationalObjectState,
    derive_conversational_objects,
)
from .contact_dynamics import ContactDynamicPoint, ContactDynamicsState, advance_contact_dynamics
from .contact_field import ContactField, ContactPoint, derive_contact_field
from .cultural_conversation_state import (
    CulturalConversationState,
    derive_cultural_conversation_state,
)
from .dot_seed import DotSeed, DotSeedSet, derive_dot_seeds
from .expressive_style_state import ExpressiveStyleState, derive_expressive_style_state
from .lightness_budget_state import LightnessBudgetState, derive_lightness_budget_state
from .ignition_loop import IgnitionLoopState, run_ignition_loop
from .interaction_effects import InteractionEffect, InteractionEffectsPlan, derive_interaction_effects
from .interaction_judgement_view import (
    InferredSignal,
    InteractionJudgementView,
    ObservedSignal,
    derive_interaction_judgement_view,
)
from .interaction_judgement_summary import (
    InteractionJudgementSummary,
    derive_interaction_judgement_summary,
)
from .interaction_judgement_comparison import (
    InteractionJudgementComparison,
    InteractionJudgementComparisonCase,
    compare_interaction_judgement_summaries,
)
from .interaction_inspection_report import (
    InteractionInspectionCaseReport,
    InteractionInspectionReport,
    build_interaction_inspection_report,
)
from .interaction_condition_report import (
    InteractionConditionReport,
    build_interaction_condition_report,
)
from .interaction_audit_bundle import (
    InteractionAuditBundle,
    build_interaction_audit_bundle,
)
from .interaction_audit_casebook import (
    InteractionAuditCaseEntry,
    build_interaction_audit_case_entry,
    select_same_utterance_audit_cases,
    update_interaction_audit_casebook,
)
from .interaction_audit_comparison import (
    InteractionAuditComparison,
    InteractionAuditComparisonCase,
    InteractionAuditMetricDifference,
    compare_interaction_audit_bundles,
)
from .interaction_audit_report import (
    InteractionAuditReport,
    build_interaction_audit_report,
)
from .group_thread_registry import (
    build_group_thread_key,
    summarize_group_thread_registry_snapshot,
    update_group_thread_registry_snapshot,
)
from .reportability_gate import ReportabilityGate, derive_reportability_gate
from .relation_competition import (
    ActiveRelationEntry,
    RelationCompetitionState,
    collect_related_person_ids,
    derive_relation_competition_state,
    summarize_person_registry_snapshot,
)
from .relational_style_memory import (
    RelationalStyleMemoryState,
    derive_relational_style_memory_state,
)
from .object_operations import ObjectOperation, ObjectOperationPlan, derive_object_operations
from .protection_mode import ProtectionModeState, derive_protection_mode
from .qualia_projector import BasicQualiaProjector, QualiaProjector, QualiaState
from .qualia_kernel_adapter import QualiaPlannerView
from .association_graph import (
    AssociationGraph,
    AssociationGraphState,
    AssociationLink,
    BasicAssociationGraph,
    apply_association_reinforcement,
    coerce_association_graph_state,
)
from .insight_event import BasicInsightDetector, InsightEvent, InsightScore
from .insight_trace import InsightTrace, derive_insight_trace
from .resonance_evaluator import (
    EstimatedOtherPersonState,
    ResonanceCandidateAssessment,
    ResonanceEvaluation,
    evaluate_interaction_resonance,
    rerank_interaction_option_candidates,
)
from .headless_runtime import HeadlessInnerOSRuntime, HeadlessTurnResult
from .scene_state import SceneState, derive_scene_state
from .social_topology_state import (
    SocialTopologyState,
    coerce_social_topology_label,
    derive_social_topology_state,
)
from .self_estimator import Estimate, EstimatorHealth, ResidualLinearSelfEstimator, SelfEstimator, evaluate_estimator_health
from .terrain_plasticity import (
    TerrainPlasticityUpdate,
    apply_terrain_plasticity,
    derive_terrain_plasticity_update,
)
from .temperament_estimate import (
    TemperamentEstimate,
    advance_temperament_traces,
    derive_temperament_estimate,
)
from .daily_carry_summary import DailyCarrySummary, DailyCarrySummaryBuilder
from .continuity_summary import ContinuitySummary, ContinuitySummaryBuilder
from .distillation_record import (
    InnerOSDistillationRecord,
    InnerOSDistillationRecordBuilder,
)
from .transfer_package import (
    InnerOSTransferPackage,
    InnerOSTransferPackageBuilder,
)
from .interaction_option_search import (
    ActionFamilyActivation,
    ActionFamilySpec,
    InteractionOptionCandidate,
    compute_action_family_activations,
    generate_interaction_option_candidates,
)

from .schemas import (
    INNER_OS_HTTP_MANIFEST_SCHEMA,
    INNER_OS_MEMORY_RECORD_SCHEMA,
    INNER_OS_RECALL_PAYLOAD_SCHEMA,
    INNER_OS_SLEEP_CONSOLIDATION_SCHEMA,
    INNER_OS_WORKING_MEMORY_SNAPSHOT_SCHEMA,
    INNER_OS_PRE_TURN_INPUT_SCHEMA,
    INNER_OS_PRE_TURN_RESULT_SCHEMA,
    INNER_OS_MEMORY_RECALL_INPUT_SCHEMA,
    INNER_OS_MEMORY_RECALL_RESULT_SCHEMA,
    INNER_OS_RESPONSE_GATE_INPUT_SCHEMA,
    INNER_OS_RESPONSE_GATE_RESULT_SCHEMA,
    INNER_OS_POST_TURN_INPUT_SCHEMA,
    INNER_OS_POST_TURN_RESULT_SCHEMA,
    INNER_OS_DISTILLATION_RECORD_SCHEMA,
    INNER_OS_TRANSFER_PACKAGE_SCHEMA,
    distillation_record_contract,
    memory_record_contract,
    recall_payload_contract,
    transfer_package_contract,
)
