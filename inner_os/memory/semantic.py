from __future__ import annotations

from dataclasses import dataclass, field

from .episodic import EpisodicRecord


@dataclass
class SemanticPattern:
    pattern_id: str
    label: str
    recurrence_weight: float = 0.0
    supporting_episode_ids: list[str] = field(default_factory=list)


def derive_semantic_hints(
    episodic_records: list[EpisodicRecord],
    *,
    min_salience: float = 0.2,
) -> list[SemanticPattern]:
    hints: list[SemanticPattern] = []
    for record in episodic_records:
        if record.salience < min_salience:
            continue
        social_relation = record.related_person_id and any(
            tag in {"social", "affiliation", "continuity"} for tag in record.tags
        )
        if social_relation:
            label = f"relation:{record.related_person_id}:{record.summary}"
        elif "continuity" in record.tags:
            label = f"continuity:{record.summary}"
        elif "danger" in record.tags:
            label = f"risk:{record.summary}"
        elif "terrain_energy" in record.tags:
            label = f"charged:{record.summary}"
        else:
            label = f"episode:{record.summary}"
        recurrence = min(1.0, record.salience + 0.08 * len(record.fixation_reasons))
        hints.append(
            SemanticPattern(
                pattern_id=f"sem:{record.episode_id}",
                label=label,
                recurrence_weight=round(recurrence, 4),
                supporting_episode_ids=[record.episode_id],
            )
        )
    return hints
