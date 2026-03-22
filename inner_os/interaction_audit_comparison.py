from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping


SECTION_LABELS = {
    "observed_lines": "観測したこと",
    "inferred_lines": "推測したこと",
    "selected_object_lines": "今回扱う対象",
    "deferred_object_lines": "今はまだ触れない対象",
    "operation_lines": "今する操作",
    "intended_effect_lines": "相手に起きてほしいこと",
    "scene_lines": "場面が効いていること",
    "relation_lines": "相手との関係が効いていること",
    "memory_lines": "記憶が効いていること",
    "integration_lines": "統合した判断",
}


@dataclass(frozen=True)
class InteractionAuditComparisonCase:
    case_id: str
    observed_lines: tuple[str, ...] = ()
    inferred_lines: tuple[str, ...] = ()
    selected_object_lines: tuple[str, ...] = ()
    deferred_object_lines: tuple[str, ...] = ()
    operation_lines: tuple[str, ...] = ()
    intended_effect_lines: tuple[str, ...] = ()
    scene_lines: tuple[str, ...] = ()
    relation_lines: tuple[str, ...] = ()
    memory_lines: tuple[str, ...] = ()
    integration_lines: tuple[str, ...] = ()
    key_metrics: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["key_metrics"] = dict(self.key_metrics or {})
        return payload


@dataclass(frozen=True)
class InteractionAuditMetricDifference:
    metric_name: str
    case_values: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "case_values": dict(self.case_values),
        }


@dataclass(frozen=True)
class InteractionAuditComparison:
    cases: tuple[InteractionAuditComparisonCase, ...] = ()
    changed_sections: tuple[str, ...] = ()
    metric_differences: tuple[InteractionAuditMetricDifference, ...] = ()
    report_lines: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cases": [item.to_dict() for item in self.cases],
            "changed_sections": list(self.changed_sections),
            "metric_differences": [item.to_dict() for item in self.metric_differences],
            "report_lines": list(self.report_lines),
        }


def compare_interaction_audit_bundles(
    bundles_by_case: Mapping[str, Mapping[str, Any]] | None,
) -> InteractionAuditComparison:
    payload = dict(bundles_by_case or {})
    ordered_cases: list[InteractionAuditComparisonCase] = []
    for case_id, bundle in payload.items():
        bundle_payload = dict(bundle or {})
        ordered_cases.append(
            InteractionAuditComparisonCase(
                case_id=str(case_id),
                observed_lines=_to_tuple(bundle_payload.get("observed_lines")),
                inferred_lines=_to_tuple(bundle_payload.get("inferred_lines")),
                selected_object_lines=_to_tuple(bundle_payload.get("selected_object_lines")),
                deferred_object_lines=_to_tuple(bundle_payload.get("deferred_object_lines")),
                operation_lines=_to_tuple(bundle_payload.get("operation_lines")),
                intended_effect_lines=_to_tuple(bundle_payload.get("intended_effect_lines")),
                scene_lines=_to_tuple(bundle_payload.get("scene_lines")),
                relation_lines=_to_tuple(bundle_payload.get("relation_lines")),
                memory_lines=_to_tuple(bundle_payload.get("memory_lines")),
                integration_lines=_to_tuple(bundle_payload.get("integration_lines")),
                key_metrics=dict(bundle_payload.get("key_metrics") or {}),
            )
        )

    changed_sections: list[str] = []
    report_lines: list[str] = []
    for section_name, section_label in SECTION_LABELS.items():
        unique_sets = {getattr(case, section_name) for case in ordered_cases}
        if len(unique_sets) <= 1:
            continue
        changed_sections.append(section_name)
        report_lines.append(f"{section_label} は条件によって変わっています。")
        for case in ordered_cases:
            section_lines = getattr(case, section_name)
            if not section_lines:
                continue
            report_lines.append(
                f"{case.case_id} では、{section_label} として「{section_lines[0]}」が出ています。"
            )

    metric_differences: list[InteractionAuditMetricDifference] = []
    metric_names = sorted(
        {
            metric_name
            for case in ordered_cases
            for metric_name in (case.key_metrics or {}).keys()
        }
    )
    for metric_name in metric_names:
        case_values = {
            case.case_id: (case.key_metrics or {}).get(metric_name)
            for case in ordered_cases
            if metric_name in (case.key_metrics or {})
        }
        comparable_values = {repr(value) for value in case_values.values()}
        if len(comparable_values) <= 1:
            continue
        metric_differences.append(
            InteractionAuditMetricDifference(
                metric_name=metric_name,
                case_values=case_values,
            )
        )
        rendered_values = " / ".join(
            f"{case_id}={_render_metric_value(value)}"
            for case_id, value in case_values.items()
        )
        report_lines.append(f"{metric_name} は条件によって変わっています。{rendered_values}")

    if not report_lines:
        report_lines.append("比べる case が足りないため、変化はまだ出ていません。")

    return InteractionAuditComparison(
        cases=tuple(ordered_cases),
        changed_sections=tuple(changed_sections),
        metric_differences=tuple(metric_differences),
        report_lines=tuple(report_lines),
    )


def _to_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


def _render_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
