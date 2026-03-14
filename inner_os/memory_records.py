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
class CommunityProfileTraceRecord(BaseMemoryRecord):
    kind: str = "community_profile_trace"
    provenance: str = "inner_community"


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
    elif kind == "community_profile_trace":
        record = CommunityProfileTraceRecord(**kwargs)
    else:
        record = ObservedRealRecord(**kwargs)
    normalized = record.to_record()
    normalized.update({
        "kind": record.kind,
        "schema": INNER_OS_MEMORY_RECORD_SCHEMA,
    })
    for extra_key in ("user_text", "assistant_text", "cue_text", "patterns", "benefit_score", "risk_score", "reinterpretation_mode", "reflective_tension", "social_self_pressure", "meaning_shift", "tentative_bias", "reinterpretation_summary", "environment_summary", "continuity_score", "social_grounding", "recent_strain", "caution_bias", "affiliation_bias", "belonging", "trust_bias", "attachment", "familiarity", "trust_memory", "role_alignment", "rupture_sensitivity", "social_update_strength", "identity_update_strength", "surface_policy_active", "surface_policy_level", "surface_policy_intent", "profile_scope", "culture_resonance", "community_resonance", "ritual_memory", "institutional_memory", "access_count", "last_accessed_at", "primed_weight", "reuse_trajectory", "interference_pressure", "consolidation_priority", "prospective_memory_pull"):
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
