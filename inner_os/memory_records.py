from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping, Optional

from .schemas import INNER_OS_MEMORY_RECORD_SCHEMA


@dataclass
class BaseMemoryRecord:
    kind: str = "observed_real"
    summary: str = ""
    text: str = ""
    memory_anchor: str = ""
    provenance: str = "lived"
    schema: str = INNER_OS_MEMORY_RECORD_SCHEMA
    culture_id: Optional[str] = None
    community_id: Optional[str] = None
    social_role: Optional[str] = None
    source_episode_id: Optional[str] = None
    confidence: Optional[float] = None
    policy_hint: Optional[str] = None

    def to_record(self) -> Dict[str, Any]:
        payload = asdict(self)
        return {k: v for k, v in payload.items() if v not in (None, "")}


@dataclass
class ObservedRealRecord(BaseMemoryRecord):
    kind: str = "observed_real"
    provenance: str = "lived"


@dataclass
class ReconstructedRecord(BaseMemoryRecord):
    kind: str = "reconstructed"
    provenance: str = "reconstruction"


@dataclass
class VerifiedRecord(BaseMemoryRecord):
    kind: str = "verified"
    provenance: str = "verification"


@dataclass
class ExperiencedSimRecord(BaseMemoryRecord):
    kind: str = "experienced_sim"
    provenance: str = "simulation"


@dataclass
class TransferredLearningRecord(BaseMemoryRecord):
    kind: str = "transferred_learning"
    provenance: str = "simulation_transfer"


@dataclass
class IdentityTraceRecord(BaseMemoryRecord):
    kind: str = "identity_trace"
    provenance: str = "inner_state"


@dataclass
class RelationshipTraceRecord(BaseMemoryRecord):
    kind: str = "relationship_trace"
    provenance: str = "inner_relation"


@dataclass
class GroupThreadTraceRecord(BaseMemoryRecord):
    kind: str = "group_thread_trace"
    provenance: str = "inner_relation"


@dataclass
class CommunityProfileTraceRecord(BaseMemoryRecord):
    kind: str = "community_profile_trace"
    provenance: str = "inner_community"


@dataclass
class ContextShiftTraceRecord(BaseMemoryRecord):
    kind: str = "context_shift_trace"
    provenance: str = "inner_context"


@dataclass
class WorkingMemoryTraceRecord(BaseMemoryRecord):
    kind: str = "working_memory_trace"
    provenance: str = "inner_working_memory"


@dataclass
class InsightTraceRecord(BaseMemoryRecord):
    kind: str = "insight_trace"
    provenance: str = "inner_insight"


@dataclass
class CommitmentTraceRecord(BaseMemoryRecord):
    kind: str = "commitment_trace"
    provenance: str = "inner_commitment"


@dataclass
class AgendaTraceRecord(BaseMemoryRecord):
    kind: str = "agenda_trace"
    provenance: str = "inner_agenda"


def normalize_memory_record(payload: Mapping[str, Any]) -> Dict[str, Any]:
    raw = dict(payload)
    kind = str(raw.get("kind") or "observed_real").strip().lower()
    summary = str(raw.get("summary") or raw.get("text") or "").strip()
    text = str(raw.get("text") or raw.get("summary") or "").strip()
    anchor = str(raw.get("memory_anchor") or summary or text).strip()[:160]
    kwargs = {
        "summary": summary,
        "text": text,
        "memory_anchor": anchor,
        "culture_id": _optional_text(raw.get("culture_id")),
        "community_id": _optional_text(raw.get("community_id")),
        "social_role": _optional_text(raw.get("social_role")),
        "source_episode_id": _optional_text(raw.get("source_episode_id") or raw.get("episode_id")),
        "confidence": _optional_float(raw.get("confidence")),
        "policy_hint": _optional_text(raw.get("policy_hint")),
    }
    if kind == "reconstructed":
        record = ReconstructedRecord(**kwargs)
    elif kind == "verified":
        record = VerifiedRecord(**kwargs)
    elif kind == "experienced_sim":
        record = ExperiencedSimRecord(**kwargs)
    elif kind == "transferred_learning":
        record = TransferredLearningRecord(**kwargs)
    elif kind == "identity_trace":
        record = IdentityTraceRecord(**kwargs)
    elif kind == "relationship_trace":
        record = RelationshipTraceRecord(**kwargs)
    elif kind == "group_thread_trace":
        record = GroupThreadTraceRecord(**kwargs)
    elif kind == "community_profile_trace":
        record = CommunityProfileTraceRecord(**kwargs)
    elif kind == "context_shift_trace":
        record = ContextShiftTraceRecord(**kwargs)
    elif kind == "working_memory_trace":
        record = WorkingMemoryTraceRecord(**kwargs)
    elif kind == "insight_trace":
        record = InsightTraceRecord(**kwargs)
    elif kind == "commitment_trace":
        record = CommitmentTraceRecord(**kwargs)
    elif kind == "agenda_trace":
        record = AgendaTraceRecord(**kwargs)
    else:
        record = ObservedRealRecord(**kwargs)
    normalized = record.to_record()
    normalized.update({
        "kind": record.kind,
        "schema": INNER_OS_MEMORY_RECORD_SCHEMA,
    })
    for extra_key in ("user_text", "assistant_text", "cue_text", "patterns", "benefit_score", "risk_score", "reinterpretation_mode", "reflective_tension", "social_self_pressure", "meaning_shift", "tentative_bias", "reinterpretation_summary", "environment_summary", "continuity_score", "social_grounding", "recent_strain", "caution_bias", "affiliation_bias", "belonging", "trust_bias", "attachment", "familiarity", "trust_memory", "role_alignment", "rupture_sensitivity", "social_update_strength", "identity_update_strength", "surface_policy_active", "surface_policy_level", "surface_policy_intent", "profile_scope", "culture_resonance", "community_resonance", "ritual_memory", "institutional_memory", "community_profile_pressure", "access_count", "last_accessed_at", "primed_weight", "reuse_trajectory", "interference_pressure", "consolidation_priority", "prospective_memory_pull", "transition_intensity", "place_changed", "body_state_changed", "privacy_shift", "density_shift", "terrain_transition_roughness", "terrain_observed_roughness", "roughness_level", "roughness_velocity", "roughness_momentum", "roughness_dwell", "defensive_level", "defensive_velocity", "defensive_momentum", "defensive_dwell", "focus_text", "focus_anchor", "current_focus", "unresolved_count", "open_loops", "carryover_load", "pending_meaning", "social_focus", "bodily_salience", "memory_pressure", "semantic_seed_focus", "semantic_seed_anchor", "semantic_seed_strength", "semantic_seed_recurrence", "working_memory_replay_focus", "working_memory_replay_anchor", "working_memory_replay_strength", "working_memory_replay_alignment", "working_memory_replay_reinforcement", "long_term_theme_kind", "long_term_theme_summary", "long_term_theme_alignment", "long_term_theme_reinforcement", "candidate_continuity_pull", "candidate_meaning_pull", "candidate_social_pull", "candidate_focus_hint", "candidate_anchor_hint", "related_person_id", "social_interpretation", "address_hint", "timing_hint", "stance_hint", "relation_episode_naming", "utterance_stance", "nonverbal_signature", "situation_phase", "relational_mood_signature", "interaction_alignment_score", "shared_attention_delta", "distance_mismatch", "hesitation_mismatch", "interaction_focus_now", "interaction_leave_closed_for_now", "interaction_response_action_now", "interaction_wanted_effect_on_other", "group_thread_id", "group_thread_focus", "top_person_ids", "thread_total_people", "threading_pressure", "visibility_pressure", "hierarchy_pressure", "memory_write_class", "memory_write_class_reason", "commitment_state", "commitment_target", "commitment_score", "commitment_winner_margin", "commitment_accepted_cost", "agenda_state", "agenda_reason", "agenda_score", "agenda_winner_margin", "insight_class", "association_link_key", "linked_seed_ids", "linked_seed_keys", "insight_score", "novelty", "coherence_gain", "prediction_drop", "reframed_topic", "followup_bias", "anchor_center", "anchor_dispersion", "confidence"):
        if extra_key in raw:
            normalized[extra_key] = raw[extra_key]
    return normalized


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
