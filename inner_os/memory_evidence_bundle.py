from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .schemas import INNER_OS_MEMORY_EVIDENCE_BUNDLE_SCHEMA


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


@dataclass(frozen=True)
class MemoryEvidenceItem:
    """時間整合つきの記憶証拠 1 件。"""

    evidence_id: str
    kind: str
    summary: str
    temporal_status: str = ""
    anchor: str = ""
    source_session_id: str = ""
    source_record_id: str = ""
    related_person_id: str = ""
    group_thread_id: str = ""
    valid_from: str = ""
    valid_to: str = ""
    weight: float = 0.0
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "summary": self.summary,
            "temporal_status": self.temporal_status,
            "anchor": self.anchor,
            "source_session_id": self.source_session_id,
            "source_record_id": self.source_record_id,
            "related_person_id": self.related_person_id,
            "group_thread_id": self.group_thread_id,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "weight": round(float(self.weight), 4),
            "tags": list(self.tags),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MemoryEvidenceItem":
        return cls(
            evidence_id=_text(payload.get("evidence_id")),
            kind=_text(payload.get("kind")),
            summary=_text(payload.get("summary")),
            temporal_status=_text(payload.get("temporal_status")),
            anchor=_text(payload.get("anchor")),
            source_session_id=_text(payload.get("source_session_id")),
            source_record_id=_text(payload.get("source_record_id")),
            related_person_id=_text(payload.get("related_person_id")),
            group_thread_id=_text(payload.get("group_thread_id")),
            valid_from=_text(payload.get("valid_from")),
            valid_to=_text(payload.get("valid_to")),
            weight=_float01(payload.get("weight")),
            tags=tuple(_text(item) for item in payload.get("tags") or () if _text(item)),
        )


@dataclass(frozen=True)
class TemporalConstraint:
    """時間検索の制約や優先条件。"""

    kind: str
    summary: str
    focus: str = ""
    priority: float = 0.0
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "summary": self.summary,
            "focus": self.focus,
            "priority": round(float(self.priority), 4),
            "source": self.source,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TemporalConstraint":
        return cls(
            kind=_text(payload.get("kind")),
            summary=_text(payload.get("summary")),
            focus=_text(payload.get("focus")),
            priority=_float01(payload.get("priority")),
            source=_text(payload.get("source")),
        )


@dataclass(frozen=True)
class ReentryContext:
    """再開すべき場や窓の候補。"""

    window: str
    summary: str
    related_person_id: str = ""
    group_thread_id: str = ""
    culture_id: str = ""
    community_id: str = ""
    priority: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "summary": self.summary,
            "related_person_id": self.related_person_id,
            "group_thread_id": self.group_thread_id,
            "culture_id": self.culture_id,
            "community_id": self.community_id,
            "priority": round(float(self.priority), 4),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ReentryContext":
        return cls(
            window=_text(payload.get("window")),
            summary=_text(payload.get("summary")),
            related_person_id=_text(payload.get("related_person_id")),
            group_thread_id=_text(payload.get("group_thread_id")),
            culture_id=_text(payload.get("culture_id")),
            community_id=_text(payload.get("community_id")),
            priority=_float01(payload.get("priority")),
        )


@dataclass(frozen=True)
class MemoryEvidenceBundle:
    """時間整合済みの recall sidecar 束。"""

    cue_text: str = ""
    focus: str = ""
    schema: str = INNER_OS_MEMORY_EVIDENCE_BUNDLE_SCHEMA
    facts_current: tuple[MemoryEvidenceItem, ...] = ()
    facts_superseded: tuple[MemoryEvidenceItem, ...] = ()
    timeline_events: tuple[MemoryEvidenceItem, ...] = ()
    temporal_constraints: tuple[TemporalConstraint, ...] = ()
    reentry_contexts: tuple[ReentryContext, ...] = ()
    source_refs: tuple[str, ...] = ()
    ambiguity_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "cue_text": self.cue_text,
            "focus": self.focus,
            "facts_current": [item.to_dict() for item in self.facts_current],
            "facts_superseded": [item.to_dict() for item in self.facts_superseded],
            "timeline_events": [item.to_dict() for item in self.timeline_events],
            "temporal_constraints": [item.to_dict() for item in self.temporal_constraints],
            "reentry_contexts": [item.to_dict() for item in self.reentry_contexts],
            "source_refs": list(self.source_refs),
            "ambiguity_notes": list(self.ambiguity_notes),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MemoryEvidenceBundle":
        return cls(
            cue_text=_text(payload.get("cue_text")),
            focus=_text(payload.get("focus")),
            schema=_text(payload.get("schema")) or INNER_OS_MEMORY_EVIDENCE_BUNDLE_SCHEMA,
            facts_current=_coerce_evidence_items(payload.get("facts_current")),
            facts_superseded=_coerce_evidence_items(payload.get("facts_superseded")),
            timeline_events=_coerce_evidence_items(payload.get("timeline_events")),
            temporal_constraints=_coerce_temporal_constraints(payload.get("temporal_constraints")),
            reentry_contexts=_coerce_reentry_contexts(payload.get("reentry_contexts")),
            source_refs=tuple(_text(item) for item in payload.get("source_refs") or () if _text(item)),
            ambiguity_notes=tuple(
                _text(item) for item in payload.get("ambiguity_notes") or () if _text(item)
            ),
        )


def build_memory_evidence_bundle(
    *,
    cue_text: str = "",
    focus: str = "",
    facts_current: Sequence[MemoryEvidenceItem | Mapping[str, Any]] = (),
    facts_superseded: Sequence[MemoryEvidenceItem | Mapping[str, Any]] = (),
    timeline_events: Sequence[MemoryEvidenceItem | Mapping[str, Any]] = (),
    temporal_constraints: Sequence[TemporalConstraint | Mapping[str, Any]] = (),
    reentry_contexts: Sequence[ReentryContext | Mapping[str, Any]] = (),
    source_refs: Sequence[str] = (),
    ambiguity_notes: Sequence[str] = (),
) -> MemoryEvidenceBundle:
    return MemoryEvidenceBundle(
        cue_text=_text(cue_text),
        focus=_text(focus),
        facts_current=_coerce_evidence_items(facts_current),
        facts_superseded=_coerce_evidence_items(facts_superseded),
        timeline_events=_coerce_evidence_items(timeline_events),
        temporal_constraints=_coerce_temporal_constraints(temporal_constraints),
        reentry_contexts=_coerce_reentry_contexts(reentry_contexts),
        source_refs=tuple(_text(item) for item in source_refs if _text(item)),
        ambiguity_notes=tuple(_text(item) for item in ambiguity_notes if _text(item)),
    )


def _coerce_evidence_items(
    values: Sequence[MemoryEvidenceItem | Mapping[str, Any]] | Any,
) -> tuple[MemoryEvidenceItem, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    items: list[MemoryEvidenceItem] = []
    for value in values:
        if isinstance(value, MemoryEvidenceItem):
            item = value
        elif isinstance(value, Mapping):
            item = MemoryEvidenceItem.from_mapping(value)
        else:
            continue
        if item.evidence_id and item.summary:
            items.append(item)
    return tuple(items)


def _coerce_temporal_constraints(
    values: Sequence[TemporalConstraint | Mapping[str, Any]] | Any,
) -> tuple[TemporalConstraint, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    items: list[TemporalConstraint] = []
    for value in values:
        if isinstance(value, TemporalConstraint):
            item = value
        elif isinstance(value, Mapping):
            item = TemporalConstraint.from_mapping(value)
        else:
            continue
        if item.kind and item.summary:
            items.append(item)
    return tuple(items)


def _coerce_reentry_contexts(
    values: Sequence[ReentryContext | Mapping[str, Any]] | Any,
) -> tuple[ReentryContext, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    items: list[ReentryContext] = []
    for value in values:
        if isinstance(value, ReentryContext):
            item = value
        elif isinstance(value, Mapping):
            item = ReentryContext.from_mapping(value)
        else:
            continue
        if item.window and item.summary:
            items.append(item)
    return tuple(items)
