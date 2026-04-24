from __future__ import annotations

from collections.abc import Iterator, Mapping as MappingABC
from dataclasses import dataclass, field
from typing import Any, Mapping


def _export_bundle_value(value: object) -> object:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _export_bundle_value(to_dict())
    if isinstance(value, Mapping):
        return {
            str(key): _export_bundle_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_export_bundle_value(item) for item in value]
    if isinstance(value, tuple):
        return [_export_bundle_value(item) for item in value]
    return value


@dataclass(frozen=True)
class QualiaHintBundleContract(MappingABC[str, object]):
    qualia_state: Mapping[str, Any] = field(default_factory=dict)
    qualia_estimator_health: Mapping[str, Any] = field(default_factory=dict)
    qualia_protection_grad_x: list[object] = field(default_factory=list)
    qualia_axis_labels: list[str] = field(default_factory=list)
    qualia_planner_view: Mapping[str, Any] = field(default_factory=dict)
    qualia_hint_source: str = "none"
    qualia_hint_version: int = 0
    qualia_hint_fallback_reason: str = ""
    qualia_hint_expected_source: str = ""
    qualia_hint_expected_mismatch: bool = False
    extras: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "qualia_state": _export_bundle_value(self.qualia_state),
            "qualia_estimator_health": _export_bundle_value(
                self.qualia_estimator_health
            ),
            "qualia_protection_grad_x": _export_bundle_value(
                self.qualia_protection_grad_x
            ),
            "qualia_axis_labels": list(self.qualia_axis_labels),
            "qualia_planner_view": _export_bundle_value(self.qualia_planner_view),
            "qualia_hint_source": self.qualia_hint_source,
            "qualia_hint_version": self.qualia_hint_version,
            "qualia_hint_fallback_reason": self.qualia_hint_fallback_reason,
            "qualia_hint_expected_source": self.qualia_hint_expected_source,
            "qualia_hint_expected_mismatch": self.qualia_hint_expected_mismatch,
        }
        data.update(self.extras)
        return data

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class InteractionAuditHintBundleContract(MappingABC[str, object]):
    interaction_audit_bundle: Mapping[str, Any] = field(default_factory=dict)
    interaction_audit_casebook: Mapping[str, Any] = field(default_factory=dict)
    interaction_audit_report: Mapping[str, Any] = field(default_factory=dict)
    interaction_audit_reference_case_ids: list[str] = field(default_factory=list)
    interaction_audit_reference_case_meta: Mapping[str, Any] = field(
        default_factory=dict
    )
    extras: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "interaction_audit_bundle": _export_bundle_value(
                self.interaction_audit_bundle
            ),
            "interaction_audit_casebook": _export_bundle_value(
                self.interaction_audit_casebook
            ),
            "interaction_audit_report": _export_bundle_value(
                self.interaction_audit_report
            ),
            "interaction_audit_reference_case_ids": list(
                self.interaction_audit_reference_case_ids
            ),
            "interaction_audit_reference_case_meta": _export_bundle_value(
                self.interaction_audit_reference_case_meta
            ),
        }
        data.update(self.extras)
        return data

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class SceneHintBundleContract(MappingABC[str, object]):
    scene_state: Mapping[str, Any] = field(default_factory=dict)
    scene_family: str = ""
    interaction_option_candidates: list[object] = field(default_factory=list)
    top_interaction_option_family: str = ""
    extras: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "scene_state": _export_bundle_value(self.scene_state),
            "scene_family": self.scene_family,
            "interaction_option_candidates": _export_bundle_value(
                self.interaction_option_candidates
            ),
            "top_interaction_option_family": self.top_interaction_option_family,
        }
        data.update(self.extras)
        return data

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class WorkspaceHintBundleContract(MappingABC[str, object]):
    conscious_workspace: Mapping[str, Any] = field(default_factory=dict)
    conscious_workspace_mode: str = ""
    conscious_workspace_reportable_slice: list[str] = field(default_factory=list)
    conscious_workspace_withheld_slice: list[str] = field(default_factory=list)
    conscious_workspace_actionable_slice: list[str] = field(default_factory=list)
    conscious_workspace_ignition_phase: str = ""
    conscious_workspace_slot_scores: Mapping[str, Any] = field(default_factory=dict)
    conscious_workspace_winner_margin: float = 0.0
    conscious_workspace_dominant_inputs: list[str] = field(default_factory=list)
    extras: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "conscious_workspace": _export_bundle_value(self.conscious_workspace),
            "conscious_workspace_mode": self.conscious_workspace_mode,
            "conscious_workspace_reportable_slice": list(
                self.conscious_workspace_reportable_slice
            ),
            "conscious_workspace_withheld_slice": list(
                self.conscious_workspace_withheld_slice
            ),
            "conscious_workspace_actionable_slice": list(
                self.conscious_workspace_actionable_slice
            ),
            "conscious_workspace_ignition_phase": self.conscious_workspace_ignition_phase,
            "conscious_workspace_slot_scores": _export_bundle_value(
                self.conscious_workspace_slot_scores
            ),
            "conscious_workspace_winner_margin": self.conscious_workspace_winner_margin,
            "conscious_workspace_dominant_inputs": list(
                self.conscious_workspace_dominant_inputs
            ),
        }
        data.update(self.extras)
        return data

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class InteractionReasoningHintBundleContract(MappingABC[str, object]):
    conversational_objects: Mapping[str, Any] = field(default_factory=dict)
    conversational_object_labels: list[str] = field(default_factory=list)
    conversational_object_pressure_balance: float = 0.0
    object_operations: Mapping[str, Any] = field(default_factory=dict)
    object_operation_question_budget: int = 0
    object_operation_question_pressure: float = 0.0
    object_operation_defer_dominance: float = 0.0
    interaction_effects: Mapping[str, Any] = field(default_factory=dict)
    interaction_judgement_view: Mapping[str, Any] = field(default_factory=dict)
    interaction_judgement_summary: Mapping[str, Any] = field(default_factory=dict)
    interaction_condition_report: Mapping[str, Any] = field(default_factory=dict)
    interaction_inspection_report: Mapping[str, Any] = field(default_factory=dict)
    extras: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "conversational_objects": _export_bundle_value(
                self.conversational_objects
            ),
            "conversational_object_labels": list(
                self.conversational_object_labels
            ),
            "conversational_object_pressure_balance": self.conversational_object_pressure_balance,
            "object_operations": _export_bundle_value(self.object_operations),
            "object_operation_question_budget": self.object_operation_question_budget,
            "object_operation_question_pressure": self.object_operation_question_pressure,
            "object_operation_defer_dominance": self.object_operation_defer_dominance,
            "interaction_effects": _export_bundle_value(self.interaction_effects),
            "interaction_judgement_view": _export_bundle_value(
                self.interaction_judgement_view
            ),
            "interaction_judgement_summary": _export_bundle_value(
                self.interaction_judgement_summary
            ),
            "interaction_condition_report": _export_bundle_value(
                self.interaction_condition_report
            ),
            "interaction_inspection_report": _export_bundle_value(
                self.interaction_inspection_report
            ),
        }
        data.update(self.extras)
        return data

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class FieldRegulationHintBundleContract(MappingABC[str, object]):
    contact_field: Mapping[str, Any] = field(default_factory=dict)
    contact_dynamics: Mapping[str, Any] = field(default_factory=dict)
    contact_dynamics_mode: str = ""
    contact_reflection_state: Mapping[str, Any] = field(default_factory=dict)
    access_projection: Mapping[str, Any] = field(default_factory=dict)
    access_dynamics: Mapping[str, Any] = field(default_factory=dict)
    access_dynamics_mode: str = ""
    affect_blend_state: Mapping[str, Any] = field(default_factory=dict)
    constraint_field: Mapping[str, Any] = field(default_factory=dict)
    extras: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "contact_field": _export_bundle_value(self.contact_field),
            "contact_dynamics": _export_bundle_value(self.contact_dynamics),
            "contact_dynamics_mode": self.contact_dynamics_mode,
            "contact_reflection_state": _export_bundle_value(
                self.contact_reflection_state
            ),
            "access_projection": _export_bundle_value(self.access_projection),
            "access_dynamics": _export_bundle_value(self.access_dynamics),
            "access_dynamics_mode": self.access_dynamics_mode,
            "affect_blend_state": _export_bundle_value(self.affect_blend_state),
            "constraint_field": _export_bundle_value(self.constraint_field),
        }
        data.update(self.extras)
        return data

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class TerrainInsightHintBundleContract(MappingABC[str, object]):
    affective_position: Mapping[str, Any] = field(default_factory=dict)
    affective_position_confidence: float = 0.0
    affective_terrain_state: Mapping[str, Any] = field(default_factory=dict)
    terrain_readout: Mapping[str, Any] = field(default_factory=dict)
    terrain_active_patch_label: str = ""
    protection_mode: Mapping[str, Any] = field(default_factory=dict)
    protection_mode_name: str = ""
    protection_mode_strength: float = 0.0
    association_graph: Mapping[str, Any] = field(default_factory=dict)
    association_graph_winner_margin: float = 0.0
    association_graph_dominant_inputs: list[str] = field(default_factory=list)
    insight_event: Mapping[str, Any] = field(default_factory=dict)
    resonance_evaluation: Mapping[str, Any] = field(default_factory=dict)
    extras: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "affective_position": _export_bundle_value(self.affective_position),
            "affective_position_confidence": self.affective_position_confidence,
            "affective_terrain_state": _export_bundle_value(
                self.affective_terrain_state
            ),
            "terrain_readout": _export_bundle_value(self.terrain_readout),
            "terrain_active_patch_label": self.terrain_active_patch_label,
            "protection_mode": _export_bundle_value(self.protection_mode),
            "protection_mode_name": self.protection_mode_name,
            "protection_mode_strength": self.protection_mode_strength,
            "association_graph": _export_bundle_value(self.association_graph),
            "association_graph_winner_margin": self.association_graph_winner_margin,
            "association_graph_dominant_inputs": list(
                self.association_graph_dominant_inputs
            ),
            "insight_event": _export_bundle_value(self.insight_event),
            "resonance_evaluation": _export_bundle_value(
                self.resonance_evaluation
            ),
        }
        data.update(self.extras)
        return data

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def coerce_qualia_hint_bundle_contract(
    value: Mapping[str, Any] | QualiaHintBundleContract | None,
) -> QualiaHintBundleContract:
    if isinstance(value, QualiaHintBundleContract):
        return value
    packet = dict(value or {})
    core_keys = {
        "qualia_state",
        "qualia_estimator_health",
        "qualia_protection_grad_x",
        "qualia_axis_labels",
        "qualia_planner_view",
        "qualia_hint_source",
        "qualia_hint_version",
        "qualia_hint_fallback_reason",
        "qualia_hint_expected_source",
        "qualia_hint_expected_mismatch",
    }
    extras = {key: packet[key] for key in packet.keys() - core_keys}
    return QualiaHintBundleContract(
        qualia_state=dict(packet.get("qualia_state") or {}),
        qualia_estimator_health=dict(packet.get("qualia_estimator_health") or {}),
        qualia_protection_grad_x=list(packet.get("qualia_protection_grad_x") or []),
        qualia_axis_labels=[
            str(item).strip()
            for item in packet.get("qualia_axis_labels") or []
            if str(item).strip()
        ],
        qualia_planner_view=dict(packet.get("qualia_planner_view") or {}),
        qualia_hint_source=str(packet.get("qualia_hint_source") or "none"),
        qualia_hint_version=int(packet.get("qualia_hint_version") or 0),
        qualia_hint_fallback_reason=str(
            packet.get("qualia_hint_fallback_reason") or ""
        ),
        qualia_hint_expected_source=str(
            packet.get("qualia_hint_expected_source") or ""
        ),
        qualia_hint_expected_mismatch=bool(
            packet.get("qualia_hint_expected_mismatch", False)
        ),
        extras=extras,
    )


def coerce_interaction_audit_hint_bundle_contract(
    value: Mapping[str, Any] | InteractionAuditHintBundleContract | None,
) -> InteractionAuditHintBundleContract:
    if isinstance(value, InteractionAuditHintBundleContract):
        return value
    packet = dict(value or {})
    core_keys = {
        "interaction_audit_bundle",
        "interaction_audit_casebook",
        "interaction_audit_report",
        "interaction_audit_reference_case_ids",
        "interaction_audit_reference_case_meta",
    }
    extras = {key: packet[key] for key in packet.keys() - core_keys}
    return InteractionAuditHintBundleContract(
        interaction_audit_bundle=dict(packet.get("interaction_audit_bundle") or {}),
        interaction_audit_casebook=dict(packet.get("interaction_audit_casebook") or {}),
        interaction_audit_report=dict(packet.get("interaction_audit_report") or {}),
        interaction_audit_reference_case_ids=[
            str(item).strip()
            for item in packet.get("interaction_audit_reference_case_ids") or []
            if str(item).strip()
        ],
        interaction_audit_reference_case_meta=dict(
            packet.get("interaction_audit_reference_case_meta") or {}
        ),
        extras=extras,
    )


def coerce_scene_hint_bundle_contract(
    value: Mapping[str, Any] | SceneHintBundleContract | None,
) -> SceneHintBundleContract:
    if isinstance(value, SceneHintBundleContract):
        return value
    packet = dict(value or {})
    core_keys = {
        "scene_state",
        "scene_family",
        "interaction_option_candidates",
        "top_interaction_option_family",
    }
    extras = {key: packet[key] for key in packet.keys() - core_keys}
    return SceneHintBundleContract(
        scene_state=dict(packet.get("scene_state") or {}),
        scene_family=str(packet.get("scene_family") or ""),
        interaction_option_candidates=list(
            packet.get("interaction_option_candidates") or []
        ),
        top_interaction_option_family=str(
            packet.get("top_interaction_option_family") or ""
        ),
        extras=extras,
    )


def coerce_workspace_hint_bundle_contract(
    value: Mapping[str, Any] | WorkspaceHintBundleContract | None,
) -> WorkspaceHintBundleContract:
    if isinstance(value, WorkspaceHintBundleContract):
        return value
    packet = dict(value or {})
    core_keys = {
        "conscious_workspace",
        "conscious_workspace_mode",
        "conscious_workspace_reportable_slice",
        "conscious_workspace_withheld_slice",
        "conscious_workspace_actionable_slice",
        "conscious_workspace_ignition_phase",
        "conscious_workspace_slot_scores",
        "conscious_workspace_winner_margin",
        "conscious_workspace_dominant_inputs",
    }
    extras = {key: packet[key] for key in packet.keys() - core_keys}
    return WorkspaceHintBundleContract(
        conscious_workspace=dict(packet.get("conscious_workspace") or {}),
        conscious_workspace_mode=str(packet.get("conscious_workspace_mode") or ""),
        conscious_workspace_reportable_slice=[
            str(item).strip()
            for item in packet.get("conscious_workspace_reportable_slice") or []
            if str(item).strip()
        ],
        conscious_workspace_withheld_slice=[
            str(item).strip()
            for item in packet.get("conscious_workspace_withheld_slice") or []
            if str(item).strip()
        ],
        conscious_workspace_actionable_slice=[
            str(item).strip()
            for item in packet.get("conscious_workspace_actionable_slice") or []
            if str(item).strip()
        ],
        conscious_workspace_ignition_phase=str(
            packet.get("conscious_workspace_ignition_phase") or ""
        ),
        conscious_workspace_slot_scores=dict(
            packet.get("conscious_workspace_slot_scores") or {}
        ),
        conscious_workspace_winner_margin=float(
            packet.get("conscious_workspace_winner_margin") or 0.0
        ),
        conscious_workspace_dominant_inputs=[
            str(item).strip()
            for item in packet.get("conscious_workspace_dominant_inputs") or []
            if str(item).strip()
        ],
        extras=extras,
    )


def coerce_interaction_reasoning_hint_bundle_contract(
    value: Mapping[str, Any] | InteractionReasoningHintBundleContract | None,
) -> InteractionReasoningHintBundleContract:
    if isinstance(value, InteractionReasoningHintBundleContract):
        return value
    packet = dict(value or {})
    core_keys = {
        "conversational_objects",
        "conversational_object_labels",
        "conversational_object_pressure_balance",
        "object_operations",
        "object_operation_question_budget",
        "object_operation_question_pressure",
        "object_operation_defer_dominance",
        "interaction_effects",
        "interaction_judgement_view",
        "interaction_judgement_summary",
        "interaction_condition_report",
        "interaction_inspection_report",
    }
    extras = {key: packet[key] for key in packet.keys() - core_keys}
    return InteractionReasoningHintBundleContract(
        conversational_objects=dict(packet.get("conversational_objects") or {}),
        conversational_object_labels=[
            str(item).strip()
            for item in packet.get("conversational_object_labels") or []
            if str(item).strip()
        ],
        conversational_object_pressure_balance=float(
            packet.get("conversational_object_pressure_balance") or 0.0
        ),
        object_operations=dict(packet.get("object_operations") or {}),
        object_operation_question_budget=int(
            packet.get("object_operation_question_budget") or 0
        ),
        object_operation_question_pressure=float(
            packet.get("object_operation_question_pressure") or 0.0
        ),
        object_operation_defer_dominance=float(
            packet.get("object_operation_defer_dominance") or 0.0
        ),
        interaction_effects=dict(packet.get("interaction_effects") or {}),
        interaction_judgement_view=dict(
            packet.get("interaction_judgement_view") or {}
        ),
        interaction_judgement_summary=dict(
            packet.get("interaction_judgement_summary") or {}
        ),
        interaction_condition_report=dict(
            packet.get("interaction_condition_report") or {}
        ),
        interaction_inspection_report=dict(
            packet.get("interaction_inspection_report") or {}
        ),
        extras=extras,
    )


def coerce_field_regulation_hint_bundle_contract(
    value: Mapping[str, Any] | FieldRegulationHintBundleContract | None,
) -> FieldRegulationHintBundleContract:
    if isinstance(value, FieldRegulationHintBundleContract):
        return value
    packet = dict(value or {})
    core_keys = {
        "contact_field",
        "contact_dynamics",
        "contact_dynamics_mode",
        "contact_reflection_state",
        "access_projection",
        "access_dynamics",
        "access_dynamics_mode",
        "affect_blend_state",
        "constraint_field",
    }
    extras = {key: packet[key] for key in packet.keys() - core_keys}
    return FieldRegulationHintBundleContract(
        contact_field=dict(packet.get("contact_field") or {}),
        contact_dynamics=dict(packet.get("contact_dynamics") or {}),
        contact_dynamics_mode=str(packet.get("contact_dynamics_mode") or ""),
        contact_reflection_state=dict(packet.get("contact_reflection_state") or {}),
        access_projection=dict(packet.get("access_projection") or {}),
        access_dynamics=dict(packet.get("access_dynamics") or {}),
        access_dynamics_mode=str(packet.get("access_dynamics_mode") or ""),
        affect_blend_state=dict(packet.get("affect_blend_state") or {}),
        constraint_field=dict(packet.get("constraint_field") or {}),
        extras=extras,
    )


def coerce_terrain_insight_hint_bundle_contract(
    value: Mapping[str, Any] | TerrainInsightHintBundleContract | None,
) -> TerrainInsightHintBundleContract:
    if isinstance(value, TerrainInsightHintBundleContract):
        return value
    packet = dict(value or {})
    core_keys = {
        "affective_position",
        "affective_position_confidence",
        "affective_terrain_state",
        "terrain_readout",
        "terrain_active_patch_label",
        "protection_mode",
        "protection_mode_name",
        "protection_mode_strength",
        "association_graph",
        "association_graph_winner_margin",
        "association_graph_dominant_inputs",
        "insight_event",
        "resonance_evaluation",
    }
    extras = {key: packet[key] for key in packet.keys() - core_keys}
    return TerrainInsightHintBundleContract(
        affective_position=dict(packet.get("affective_position") or {}),
        affective_position_confidence=float(
            packet.get("affective_position_confidence") or 0.0
        ),
        affective_terrain_state=dict(packet.get("affective_terrain_state") or {}),
        terrain_readout=dict(packet.get("terrain_readout") or {}),
        terrain_active_patch_label=str(packet.get("terrain_active_patch_label") or ""),
        protection_mode=dict(packet.get("protection_mode") or {}),
        protection_mode_name=str(packet.get("protection_mode_name") or ""),
        protection_mode_strength=float(packet.get("protection_mode_strength") or 0.0),
        association_graph=dict(packet.get("association_graph") or {}),
        association_graph_winner_margin=float(
            packet.get("association_graph_winner_margin") or 0.0
        ),
        association_graph_dominant_inputs=[
            str(item).strip()
            for item in packet.get("association_graph_dominant_inputs") or []
            if str(item).strip()
        ],
        insight_event=dict(packet.get("insight_event") or {}),
        resonance_evaluation=dict(packet.get("resonance_evaluation") or {}),
        extras=extras,
    )
