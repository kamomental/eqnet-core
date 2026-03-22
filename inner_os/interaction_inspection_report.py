from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

from .interaction_judgement_comparison import (
    SECTION_LABELS,
    compare_interaction_judgement_summaries,
)


@dataclass(frozen=True)
class InteractionInspectionCaseReport:
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
class InteractionInspectionReport:
    shared_observed_lines: tuple[str, ...] = ()
    changed_sections: tuple[str, ...] = ()
    case_reports: tuple[InteractionInspectionCaseReport, ...] = ()
    report_lines: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shared_observed_lines": list(self.shared_observed_lines),
            "changed_sections": list(self.changed_sections),
            "case_reports": [item.to_dict() for item in self.case_reports],
            "report_lines": list(self.report_lines),
        }


def build_interaction_inspection_report(
    summaries_by_case: Mapping[str, Mapping[str, Any]] | None,
) -> InteractionInspectionReport:
    summaries_payload = dict(summaries_by_case or {})
    comparison = compare_interaction_judgement_summaries(summaries_payload)
    case_reports = tuple(
        InteractionInspectionCaseReport(
            case_id=case.case_id,
            observed_lines=case.observed_lines,
            inferred_lines=case.inferred_lines,
            selected_object_lines=case.selected_object_lines,
            deferred_object_lines=case.deferred_object_lines,
            operation_lines=case.operation_lines,
            intended_effect_lines=case.intended_effect_lines,
        )
        for case in comparison.cases
    )
    shared_observed_lines = _shared_lines(summaries_payload, "observed_lines")

    report_lines: list[str] = []
    for line in shared_observed_lines:
        report_lines.append(f"共通して観測したこと: {line}")

    if len(case_reports) == 1:
        case = case_reports[0]
        report_lines.extend(_single_case_lines(case))

    for section_name in comparison.changed_sections:
        section_label = SECTION_LABELS.get(section_name, section_name)
        report_lines.append(f"{section_label} は条件によって変わっています。")
        for case in case_reports:
            lines = getattr(case, section_name)
            if not lines:
                continue
            report_lines.append(
                f"{case.case_id} では「{section_label}」として「{lines[0]}」が出ています。"
            )

    return InteractionInspectionReport(
        shared_observed_lines=shared_observed_lines,
        changed_sections=comparison.changed_sections,
        case_reports=case_reports,
        report_lines=tuple(report_lines),
    )


def _shared_lines(
    summaries_payload: Mapping[str, Mapping[str, Any]],
    section_name: str,
) -> tuple[str, ...]:
    normalized_sections = []
    for summary in summaries_payload.values():
        payload = dict(summary or {})
        lines = tuple(
            str(item) for item in payload.get(section_name) or [] if str(item).strip()
        )
        normalized_sections.append(lines)
    if not normalized_sections:
        return ()
    shared = set(normalized_sections[0])
    for lines in normalized_sections[1:]:
        shared &= set(lines)
    return tuple(line for line in normalized_sections[0] if line in shared)


def _single_case_lines(case: InteractionInspectionCaseReport) -> list[str]:
    lines: list[str] = [f"{case.case_id} の確認結果です。"]
    lines.extend(f"観測したこと: {line}" for line in case.observed_lines)
    lines.extend(f"推測したこと: {line}" for line in case.inferred_lines)
    lines.extend(f"今回扱う対象: {line}" for line in case.selected_object_lines)
    lines.extend(f"今回はまだ触れない対象: {line}" for line in case.deferred_object_lines)
    lines.extend(f"今する操作: {line}" for line in case.operation_lines)
    lines.extend(f"相手に起きてほしいこと: {line}" for line in case.intended_effect_lines)
    return lines
