from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from .interaction_audit_comparison import (
    SECTION_LABELS,
    compare_interaction_audit_bundles,
)


@dataclass(frozen=True)
class InteractionAuditReport:
    changed_sections: tuple[str, ...] = ()
    metric_differences: tuple[Dict[str, Any], ...] = ()
    report_lines: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "changed_sections": list(self.changed_sections),
            "metric_differences": [dict(item) for item in self.metric_differences],
            "report_lines": list(self.report_lines),
        }


def build_interaction_audit_report(
    bundles_by_case: Mapping[str, Mapping[str, Any]] | None,
) -> InteractionAuditReport:
    bundles_payload = dict(bundles_by_case or {})
    comparison = compare_interaction_audit_bundles(bundles_payload)
    report_lines: list[str] = []

    if len(comparison.cases) == 1:
        case = comparison.cases[0]
        report_lines.extend(_single_case_lines(case))

    for section_name in comparison.changed_sections:
        label = SECTION_LABELS.get(section_name, section_name)
        report_lines.append(f"{label} は条件によって変わっています。")
        for case in comparison.cases:
            lines = getattr(case, section_name)
            if not lines:
                continue
            report_lines.append(
                f"{case.case_id} では、{label} として「{lines[0]}」が出ています。"
            )

    metric_differences = [item.to_dict() for item in comparison.metric_differences]
    for metric in comparison.metric_differences:
        rendered_values = " / ".join(
            f"{case_id}={_render_metric_value(value)}"
            for case_id, value in metric.case_values.items()
        )
        report_lines.append(f"{metric.metric_name} は条件によって変わっています。{rendered_values}")

    if not report_lines:
        report_lines.append("比べる case が足りないため、変化はまだ出ていません。")

    return InteractionAuditReport(
        changed_sections=comparison.changed_sections,
        metric_differences=tuple(metric_differences),
        report_lines=tuple(report_lines),
    )


def _render_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _single_case_lines(case: Any) -> list[str]:
    lines: list[str] = [f"{case.case_id} の監査結果です。"]
    lines.extend(f"観測したこと: {line}" for line in case.observed_lines)
    lines.extend(f"推測したこと: {line}" for line in case.inferred_lines)
    lines.extend(f"今回扱う対象: {line}" for line in case.selected_object_lines)
    lines.extend(f"今はまだ触れない対象: {line}" for line in case.deferred_object_lines)
    lines.extend(f"今する操作: {line}" for line in case.operation_lines)
    lines.extend(f"相手に起きてほしいこと: {line}" for line in case.intended_effect_lines)
    if case.scene_lines:
        lines.append(f"場面が効いていること: {case.scene_lines[0]}")
    if case.relation_lines:
        lines.append(f"相手との関係が効いていること: {case.relation_lines[0]}")
    if case.memory_lines:
        lines.append(f"記憶が効いていること: {case.memory_lines[0]}")
    if case.integration_lines:
        lines.append(f"統合した判断: {case.integration_lines[0]}")
    return lines
