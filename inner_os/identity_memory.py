from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


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


def _compact(values: Sequence[str], *, limit: int = 5) -> tuple[str, ...]:
    seen: list[str] = []
    for value in values:
        text = _text(value)
        if text and text not in seen:
            seen.append(text)
        if len(seen) >= limit:
            break
    return tuple(seen)


def _identity_arc_id(summary: Mapping[str, Any]) -> str:
    arc_kind = _text(summary.get("arc_kind")) or "continuity"
    related_person_id = _text(summary.get("related_person_id"))
    group_thread_focus = _text(summary.get("group_thread_focus"))
    theme_kind = _text(summary.get("long_term_theme_kind"))
    theme_focus = _text(summary.get("long_term_theme_focus"))
    memory_anchor = _text(summary.get("memory_anchor"))
    tokens = [
        arc_kind,
        related_person_id or group_thread_focus or "ambient",
        theme_kind or "unthemed",
        theme_focus or memory_anchor or "unanchored",
    ]
    return "::".join(token.replace(" ", "_") for token in tokens if token)


def _status_from_phase(phase: str, open_tensions: Sequence[str]) -> str:
    current_phase = _text(phase) or "forming"
    tension_set = {_text(item) for item in open_tensions if _text(item)}
    if current_phase == "integrating" and not tension_set:
        return "integrated"
    if current_phase == "shifting":
        return "active"
    if current_phase == "holding":
        return "holding"
    if "timing_sensitive_reentry" in tension_set or "multi_thread_balance" in tension_set:
        return "deferred"
    return "emerging"


@dataclass(frozen=True)
class IdentityArcRecord:
    arc_id: str
    arc_kind: str
    phase: str
    status: str
    first_day: str
    last_day: str
    days_seen: int
    update_count: int
    summary: str
    dominant_driver: str
    supporting_drivers: tuple[str, ...]
    open_tensions: tuple[str, ...]
    stability_mean: float
    stability_peak: float
    memory_anchor: str
    related_person_id: str
    group_thread_focus: str
    long_term_theme_kind: str
    long_term_theme_focus: str
    learning_mode_focus: str
    social_experiment_focus: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "arc_id": self.arc_id,
            "arc_kind": self.arc_kind,
            "phase": self.phase,
            "status": self.status,
            "first_day": self.first_day,
            "last_day": self.last_day,
            "days_seen": int(self.days_seen),
            "update_count": int(self.update_count),
            "summary": self.summary,
            "dominant_driver": self.dominant_driver,
            "supporting_drivers": list(self.supporting_drivers),
            "open_tensions": list(self.open_tensions),
            "stability_mean": round(float(self.stability_mean), 4),
            "stability_peak": round(float(self.stability_peak), 4),
            "memory_anchor": self.memory_anchor,
            "related_person_id": self.related_person_id,
            "group_thread_focus": self.group_thread_focus,
            "long_term_theme_kind": self.long_term_theme_kind,
            "long_term_theme_focus": self.long_term_theme_focus,
            "learning_mode_focus": self.learning_mode_focus,
            "social_experiment_focus": self.social_experiment_focus,
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any] | None) -> "IdentityArcRecord":
        source = dict(payload or {})
        return IdentityArcRecord(
            arc_id=_text(source.get("arc_id")),
            arc_kind=_text(source.get("arc_kind")),
            phase=_text(source.get("phase")) or "forming",
            status=_text(source.get("status")) or "emerging",
            first_day=_text(source.get("first_day")),
            last_day=_text(source.get("last_day")),
            days_seen=max(int(source.get("days_seen") or 0), 0),
            update_count=max(int(source.get("update_count") or 0), 0),
            summary=_text(source.get("summary")),
            dominant_driver=_text(source.get("dominant_driver")),
            supporting_drivers=_compact(source.get("supporting_drivers") or []),
            open_tensions=_compact(source.get("open_tensions") or []),
            stability_mean=_float01(source.get("stability_mean")),
            stability_peak=_float01(source.get("stability_peak")),
            memory_anchor=_text(source.get("memory_anchor")),
            related_person_id=_text(source.get("related_person_id")),
            group_thread_focus=_text(source.get("group_thread_focus")),
            long_term_theme_kind=_text(source.get("long_term_theme_kind")),
            long_term_theme_focus=_text(source.get("long_term_theme_focus")),
            learning_mode_focus=_text(source.get("learning_mode_focus")),
            social_experiment_focus=_text(source.get("social_experiment_focus")),
        )


class IdentityArcRegistry:
    def __init__(self, records: Sequence[IdentityArcRecord] | None = None) -> None:
        self.records: list[IdentityArcRecord] = list(records or [])

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [record.to_dict() for record in self.records],
            "summary": self.summary(),
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any] | None) -> "IdentityArcRegistry":
        source = dict(payload or {})
        records = [
            IdentityArcRecord.from_dict(item)
            for item in source.get("records") or []
            if isinstance(item, Mapping)
        ]
        return IdentityArcRegistry(records)

    def update(self, *, day_key: str, identity_arc_summary: Mapping[str, Any] | None) -> IdentityArcRecord | None:
        source = dict(identity_arc_summary or {})
        arc_kind = _text(source.get("arc_kind"))
        summary_text = _text(source.get("summary"))
        memory_anchor = _text(source.get("memory_anchor"))
        if not arc_kind and not summary_text and not memory_anchor:
            return None

        arc_id = _identity_arc_id(source)
        related_person_id = _text(source.get("related_person_id"))
        group_thread_focus = _text(source.get("group_thread_focus"))
        long_term_theme_kind = _text(source.get("long_term_theme_kind"))
        long_term_theme_focus = _text(source.get("long_term_theme_focus"))
        learning_mode_focus = _text(source.get("learning_mode_focus"))
        social_experiment_focus = _text(source.get("social_experiment_focus"))
        dominant_driver = _text(source.get("dominant_driver"))
        supporting_drivers = _compact(source.get("supporting_drivers") or [])
        open_tensions = _compact([source.get("open_tension")] + list(source.get("open_tensions") or []))
        stability = _float01(source.get("stability"))
        phase = _text(source.get("phase")) or "forming"

        existing_index = next((idx for idx, record in enumerate(self.records) if record.arc_id == arc_id), None)
        if existing_index is None:
            next_record = IdentityArcRecord(
                arc_id=arc_id,
                arc_kind=arc_kind or "continuity",
                phase=phase,
                status=_status_from_phase(phase, open_tensions),
                first_day=day_key,
                last_day=day_key,
                days_seen=1,
                update_count=1,
                summary=summary_text,
                dominant_driver=dominant_driver,
                supporting_drivers=supporting_drivers,
                open_tensions=open_tensions,
                stability_mean=stability,
                stability_peak=stability,
                memory_anchor=memory_anchor,
                related_person_id=related_person_id,
                group_thread_focus=group_thread_focus,
                long_term_theme_kind=long_term_theme_kind,
                long_term_theme_focus=long_term_theme_focus,
                learning_mode_focus=learning_mode_focus,
                social_experiment_focus=social_experiment_focus,
            )
            self.records.append(next_record)
        else:
            previous = self.records[existing_index]
            already_seen_today = previous.last_day == day_key
            total_updates = max(previous.update_count, 1)
            next_update_count = total_updates + 1
            next_mean = (
                (previous.stability_mean * total_updates) + stability
            ) / next_update_count
            merged_supporting = _compact((*previous.supporting_drivers, *supporting_drivers))
            merged_tensions = _compact((*previous.open_tensions, *open_tensions))
            next_record = IdentityArcRecord(
                arc_id=previous.arc_id,
                arc_kind=arc_kind or previous.arc_kind,
                phase=phase or previous.phase,
                status=_status_from_phase(phase or previous.phase, merged_tensions),
                first_day=previous.first_day or day_key,
                last_day=day_key,
                days_seen=previous.days_seen if already_seen_today else previous.days_seen + 1,
                update_count=next_update_count,
                summary=summary_text or previous.summary,
                dominant_driver=dominant_driver or previous.dominant_driver,
                supporting_drivers=merged_supporting or previous.supporting_drivers,
                open_tensions=merged_tensions,
                stability_mean=_float01(next_mean),
                stability_peak=max(previous.stability_peak, stability),
                memory_anchor=memory_anchor or previous.memory_anchor,
                related_person_id=related_person_id or previous.related_person_id,
                group_thread_focus=group_thread_focus or previous.group_thread_focus,
                long_term_theme_kind=long_term_theme_kind or previous.long_term_theme_kind,
                long_term_theme_focus=long_term_theme_focus or previous.long_term_theme_focus,
                learning_mode_focus=learning_mode_focus or previous.learning_mode_focus,
                social_experiment_focus=social_experiment_focus or previous.social_experiment_focus,
            )
            self.records[existing_index] = next_record

        self.records.sort(
            key=lambda record: (
                record.last_day,
                record.stability_peak,
                record.days_seen,
                record.arc_id,
            ),
            reverse=True,
        )
        return next_record

    def summary(self, *, limit: int = 3) -> dict[str, Any]:
        if not self.records:
            return {
                "dominant_arc_id": "",
                "dominant_arc_kind": "",
                "active_arc_count": 0,
                "total_arcs": 0,
                "top_arc_ids": [],
                "status_counts": {},
            }
        top_records = self.records[: max(limit, 0)]
        status_counts: dict[str, int] = {}
        for record in self.records:
            status_counts[record.status] = status_counts.get(record.status, 0) + 1
        dominant = top_records[0]
        return {
            "dominant_arc_id": dominant.arc_id,
            "dominant_arc_kind": dominant.arc_kind,
            "dominant_arc_phase": dominant.phase,
            "dominant_arc_summary": dominant.summary,
            "active_arc_count": sum(1 for record in self.records if record.status in {"active", "holding", "emerging", "deferred"}),
            "total_arcs": len(self.records),
            "top_arc_ids": [record.arc_id for record in top_records],
            "status_counts": status_counts,
            "top_arcs": [record.to_dict() for record in top_records],
        }
