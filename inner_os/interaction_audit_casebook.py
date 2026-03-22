from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class InteractionAuditCaseEntry:
    observed_text: str = ""
    condition_key: str = ""
    scene_family: str = ""
    relation_hint: str = ""
    memory_anchor: str = ""
    judgement_summary: Dict[str, Any] | None = None
    audit_bundle: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["judgement_summary"] = dict(self.judgement_summary or {})
        payload["audit_bundle"] = dict(self.audit_bundle or {})
        return payload


def build_interaction_audit_case_entry(
    *,
    observed_text: str = "",
    judgement_summary: Mapping[str, Any] | None = None,
    audit_bundle: Mapping[str, Any] | None = None,
    scene_state: Mapping[str, Any] | None = None,
    relation_context: Mapping[str, Any] | None = None,
    memory_context: Mapping[str, Any] | None = None,
) -> InteractionAuditCaseEntry:
    scene = dict(scene_state or {})
    relation = dict(relation_context or {})
    memory = dict(memory_context or {})
    scene_family = str(scene.get("scene_family") or "").strip()
    relation_hint = str(
        relation.get("partner_social_interpretation")
        or relation.get("partner_stance_hint")
        or relation.get("partner_timing_hint")
        or ""
    ).strip()
    memory_anchor = str(memory.get("memory_anchor") or "").strip()
    condition_parts = [part for part in (scene_family, relation_hint, memory_anchor) if part]
    condition_key = "|".join(condition_parts)
    return InteractionAuditCaseEntry(
        observed_text=str(observed_text or "").strip(),
        condition_key=condition_key,
        scene_family=scene_family,
        relation_hint=relation_hint,
        memory_anchor=memory_anchor,
        judgement_summary=dict(judgement_summary or {}),
        audit_bundle=dict(audit_bundle or {}),
    )


def update_interaction_audit_casebook(
    previous_casebook: Mapping[str, Any] | None,
    current_entry: InteractionAuditCaseEntry,
    *,
    max_entries: int = 6,
) -> Dict[str, Any]:
    previous_entries = _parse_case_entries(previous_casebook)
    deduped_entries = [current_entry]
    for entry in previous_entries:
        if (
            entry.observed_text == current_entry.observed_text
            and entry.condition_key == current_entry.condition_key
        ):
            continue
        deduped_entries.append(entry)
    return {
        "cases": [entry.to_dict() for entry in deduped_entries[: max(1, max_entries)]],
    }


def select_same_utterance_audit_cases(
    casebook: Mapping[str, Any] | None,
    *,
    current_text: str = "",
    max_cases: int = 2,
) -> Dict[str, Any]:
    normalized_text = str(current_text or "").strip()
    selected_summaries: Dict[str, Dict[str, Any]] = {}
    selected_bundles: Dict[str, Dict[str, Any]] = {}
    selected_meta: Dict[str, Dict[str, Any]] = {}
    selected_ids: list[str] = []
    if not normalized_text:
        return {
            "judgement_summary_cases": selected_summaries,
            "audit_bundle_cases": selected_bundles,
            "reference_case_ids": selected_ids,
            "reference_case_meta": selected_meta,
        }

    matching_entries = [
        entry
        for entry in _parse_case_entries(casebook)
        if entry.observed_text == normalized_text
    ]
    for index, entry in enumerate(matching_entries[: max(0, max_cases)], start=1):
        case_id = f"reference_{index}"
        selected_ids.append(case_id)
        selected_summaries[case_id] = dict(entry.judgement_summary or {})
        selected_bundles[case_id] = dict(entry.audit_bundle or {})
        selected_meta[case_id] = {
            "scene_family": entry.scene_family,
            "relation_hint": entry.relation_hint,
            "memory_anchor": entry.memory_anchor,
            "condition_key": entry.condition_key,
        }

    return {
        "judgement_summary_cases": selected_summaries,
        "audit_bundle_cases": selected_bundles,
        "reference_case_ids": selected_ids,
        "reference_case_meta": selected_meta,
    }


def _parse_case_entries(casebook: Mapping[str, Any] | None) -> list[InteractionAuditCaseEntry]:
    payload = dict(casebook or {})
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, (list, tuple)):
        return []
    entries: list[InteractionAuditCaseEntry] = []
    for raw_entry in raw_cases:
        if not isinstance(raw_entry, Mapping):
            continue
        entries.append(
            InteractionAuditCaseEntry(
                observed_text=str(raw_entry.get("observed_text") or "").strip(),
                condition_key=str(raw_entry.get("condition_key") or "").strip(),
                scene_family=str(raw_entry.get("scene_family") or "").strip(),
                relation_hint=str(raw_entry.get("relation_hint") or "").strip(),
                memory_anchor=str(raw_entry.get("memory_anchor") or "").strip(),
                judgement_summary=dict(raw_entry.get("judgement_summary") or {}),
                audit_bundle=dict(raw_entry.get("audit_bundle") or {}),
            )
        )
    return entries
