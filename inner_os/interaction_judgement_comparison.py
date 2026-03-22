from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping


SECTION_LABELS = {
    "observed_lines": "観測したこと",
    "inferred_lines": "推測したこと",
    "selected_object_lines": "今回扱う対象",
    "deferred_object_lines": "今回はまだ触れない対象",
    "operation_lines": "今する操作",
    "intended_effect_lines": "相手に起きてほしいこと",
}


@dataclass(frozen=True)
class InteractionJudgementComparisonCase:
    case_id: str
    observed_lines: tuple[str, ...] = ()
    inferred_lines: tuple[str, ...] = ()
    selected_object_lines: tuple[str, ...] = ()
    deferred_object_lines: tuple[str, ...] = ()
    operation_lines: tuple[str, ...] = ()
    intended_effect_lines: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InteractionJudgementComparison:
    cases: tuple[InteractionJudgementComparisonCase, ...] = ()
    changed_sections: tuple[str, ...] = ()
    difference_lines: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cases": [item.to_dict() for item in self.cases],
            "changed_sections": list(self.changed_sections),
            "difference_lines": list(self.difference_lines),
        }


def compare_interaction_judgement_summaries(
    summaries_by_case: Mapping[str, Mapping[str, Any]] | None,
) -> InteractionJudgementComparison:
    payload = dict(summaries_by_case or {})
    ordered_cases: list[InteractionJudgementComparisonCase] = []
    for case_id, summary in payload.items():
        summary_payload = dict(summary or {})
        ordered_cases.append(
            InteractionJudgementComparisonCase(
                case_id=str(case_id),
                observed_lines=_to_tuple(summary_payload.get("observed_lines")),
                inferred_lines=_to_tuple(summary_payload.get("inferred_lines")),
                selected_object_lines=_to_tuple(summary_payload.get("selected_object_lines")),
                deferred_object_lines=_to_tuple(summary_payload.get("deferred_object_lines")),
                operation_lines=_to_tuple(summary_payload.get("operation_lines")),
                intended_effect_lines=_to_tuple(summary_payload.get("intended_effect_lines")),
            )
        )

    changed_sections: list[str] = []
    difference_lines: list[str] = []
    for section_name in SECTION_LABELS:
        unique_sets = {getattr(case, section_name) for case in ordered_cases}
        if len(unique_sets) <= 1:
            continue
        changed_sections.append(section_name)
        section_label = SECTION_LABELS[section_name]
        difference_lines.append(f"{section_label} は条件によって変わっています。")
        for case in ordered_cases:
            section_lines = getattr(case, section_name)
            if not section_lines:
                continue
            difference_lines.append(
                f"{case.case_id} では「{section_label}」として「{section_lines[0]}」が出ています。"
            )

    return InteractionJudgementComparison(
        cases=tuple(ordered_cases),
        changed_sections=tuple(changed_sections),
        difference_lines=tuple(difference_lines),
    )


def _to_tuple(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(str(item) for item in value if str(item).strip())
