from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional


def memory_reference_to_record(
    memory_reference: Optional[Mapping[str, Any]],
    *,
    relational_context: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    if not isinstance(memory_reference, Mapping):
        return None
    meta = memory_reference.get("meta") if isinstance(memory_reference.get("meta"), Mapping) else {}
    candidate = memory_reference.get("candidate") if isinstance(memory_reference.get("candidate"), Mapping) else {}
    fidelity = _safe_float(memory_reference.get("fidelity"), 0.0)
    source_class = str(meta.get("source_class") or "uncertain")
    audit_event = str(meta.get("audit_event") or "")
    label = str(candidate.get("label") or meta.get("cue_label") or memory_reference.get("reply") or "").strip()
    if not label:
        return None
    kind = str(meta.get("record_kind") or "").strip() or "observed_real"
    if not kind:
        kind = "observed_real"
    if kind == "observed_real":
        if source_class == "uncertain" or audit_event in {"SOURCE_FUZZY", "DOUBLE_TAKE"}:
            kind = "reconstructed"
        elif fidelity >= 0.82 and source_class == "self":
            kind = "verified"
    return {
        "kind": kind,
        "summary": str(memory_reference.get("summary") or label),
        "text": str(memory_reference.get("reply") or memory_reference.get("text") or label),
        "memory_anchor": str(memory_reference.get("memory_anchor") or label[:160]),
        "confidence": fidelity,
        "culture_id": relational_context.get("culture_id"),
        "community_id": relational_context.get("community_id"),
        "social_role": relational_context.get("social_role"),
        "policy_hint": meta.get("memory_kind"),
        "source_episode_id": meta.get("trace_id") or meta.get("cue_label"),
        "provenance": str(meta.get("record_provenance") or "eqnet_memory_reference"),
        "surface_policy_active": relational_context.get("surface_policy_active"),
        "surface_policy_level": relational_context.get("surface_policy_level"),
        "surface_policy_intent": relational_context.get("surface_policy_intent"),
        "consolidation_priority": relational_context.get("consolidation_priority"),
        "prospective_memory_pull": relational_context.get("prospective_memory_pull"),
    }


def observed_vision_to_record(
    vision_entry: Optional[Mapping[str, Any]],
    *,
    relational_context: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    if not isinstance(vision_entry, Mapping):
        return None
    text = str(vision_entry.get("text") or vision_entry.get("summary") or "").strip()
    if not text:
        return None
    if vision_entry.get("suppressed"):
        return None
    return {
        "kind": "observed_real",
        "summary": str(vision_entry.get("summary") or text).strip(),
        "text": text,
        "memory_anchor": str(vision_entry.get("memory_anchor") or relational_context.get("place_memory_anchor") or text[:160]).strip()[:160],
        "culture_id": relational_context.get("culture_id"),
        "community_id": relational_context.get("community_id"),
        "social_role": relational_context.get("social_role"),
        "source_episode_id": vision_entry.get("id"),
        "provenance": "observed_vision",
        "surface_policy_active": relational_context.get("surface_policy_active"),
        "surface_policy_level": relational_context.get("surface_policy_level"),
        "surface_policy_intent": relational_context.get("surface_policy_intent"),
        "consolidation_priority": relational_context.get("consolidation_priority"),
        "prospective_memory_pull": relational_context.get("prospective_memory_pull"),
    }


def collect_runtime_memory_candidates(
    *,
    recall_payload: Optional[Mapping[str, Any]] = None,
    memory_reference: Optional[Mapping[str, Any]] = None,
    vision_entry: Optional[Mapping[str, Any]] = None,
    relational_context: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    if isinstance(recall_payload, Mapping) and recall_payload:
        candidates.append(dict(recall_payload))
    bridged_reference = memory_reference_to_record(
        memory_reference,
        relational_context=relational_context,
    )
    if bridged_reference:
        candidates.append(bridged_reference)
    bridged_vision = observed_vision_to_record(
        vision_entry,
        relational_context=relational_context,
    )
    if bridged_vision:
        candidates.append(bridged_vision)
    return _dedupe_candidates(candidates)


def _dedupe_candidates(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        payload = dict(record)
        key = (
            str(payload.get("kind") or ""),
            str(payload.get("memory_anchor") or payload.get("summary") or "").strip(),
            str(payload.get("provenance") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(payload)
    return deduped


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
