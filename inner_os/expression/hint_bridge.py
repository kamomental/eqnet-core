from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from ..qualia_kernel_adapter import build_qualia_planner_view

QUALIA_HINT_VERSION = 1
_SHARED_REASON = "prebuilt_shared_view"
_FALLBACK_REASON = "bridge_reconstructed_from_raw_materials"
_NONE_REASON = "neutral_no_qualia_material"
_QUALIA_HINT_KEYS = (
    "qualia_state",
    "qualia_estimator_health",
    "qualia_protection_grad_x",
    "qualia_axis_labels",
    "qualia_planner_view",
)


def build_expression_hints_from_gate_result(
    gate_result: Any,
    existing_hints: Mapping[str, object] | None = None,
    *,
    expected_source: str | None = None,
) -> dict[str, object]:
    merged = dict(existing_hints or {})
    source_hints = _extract_expression_hints(gate_result)
    for key in _QUALIA_HINT_KEYS:
        if key in source_hints:
            merged[key] = deepcopy(source_hints[key])
    return ensure_qualia_planner_view(merged, expected_source=expected_source)


def ensure_qualia_planner_view(
    expression_hints: Mapping[str, object] | None,
    *,
    expected_source: str | None = None,
) -> dict[str, object]:
    merged = dict(expression_hints or {})
    if isinstance(merged.get("qualia_planner_view"), Mapping):
        merged["qualia_planner_view"] = dict(merged["qualia_planner_view"])
        return _finalize_qualia_hint_metadata(
            merged,
            source="shared",
            reason=_SHARED_REASON,
            expected_source=expected_source,
        )

    fallback_view = build_qualia_planner_view_hint(merged)
    if fallback_view is not None:
        merged["qualia_planner_view"] = fallback_view
        return _finalize_qualia_hint_metadata(
            merged,
            source="fallback",
            reason=_FALLBACK_REASON,
            expected_source=expected_source,
        )

    merged["qualia_planner_view"] = build_qualia_planner_view(
        qualia_state=None,
        estimator_health=None,
        protection_grad_x=None,
        axis_labels=None,
    ).to_dict()
    return _finalize_qualia_hint_metadata(
        merged,
        source="none",
        reason=_NONE_REASON,
        expected_source=expected_source,
    )


def _extract_expression_hints(gate_result: Any) -> dict[str, object]:
    if isinstance(gate_result, Mapping):
        nested = gate_result.get("expression_hints")
        if isinstance(nested, Mapping):
            return dict(nested)
        return dict(gate_result)
    expression_hints = getattr(gate_result, "expression_hints", None)
    if isinstance(expression_hints, Mapping):
        return dict(expression_hints)
    return {}


def _has_qualia_raw_materials(expression_hints: Mapping[str, object]) -> bool:
    return any(
        (
            isinstance(expression_hints.get("qualia_state"), dict),
            isinstance(expression_hints.get("qualia_estimator_health"), dict),
            isinstance(expression_hints.get("qualia_protection_grad_x"), list),
            isinstance(expression_hints.get("qualia_axis_labels"), list),
        )
    )


def build_qualia_planner_view_hint(
    expression_hints: Mapping[str, object],
) -> dict[str, object] | None:
    if not _has_qualia_raw_materials(expression_hints):
        return None

    qualia_state = expression_hints.get("qualia_state")
    qualia_health = expression_hints.get("qualia_estimator_health")
    qualia_grad = expression_hints.get("qualia_protection_grad_x")
    qualia_axis_labels = expression_hints.get("qualia_axis_labels")
    return build_qualia_planner_view(
        qualia_state=qualia_state if isinstance(qualia_state, dict) else None,
        estimator_health=qualia_health if isinstance(qualia_health, dict) else None,
        protection_grad_x=qualia_grad if isinstance(qualia_grad, list) else None,
        axis_labels=qualia_axis_labels if isinstance(qualia_axis_labels, list) else None,
    ).to_dict()


def _finalize_qualia_hint_metadata(
    expression_hints: dict[str, object],
    *,
    source: str,
    reason: str,
    expected_source: str | None,
) -> dict[str, object]:
    expression_hints["qualia_hint_source"] = source
    expression_hints["qualia_hint_version"] = QUALIA_HINT_VERSION
    expression_hints["qualia_hint_fallback_reason"] = reason
    expression_hints["qualia_hint_expected_source"] = expected_source
    expression_hints["qualia_hint_expected_mismatch"] = bool(
        expected_source and source != expected_source
    )
    return expression_hints
