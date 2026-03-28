from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence


def _float01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _list_size(value: Any) -> float:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return float(len(value))
    return 0.0


def _merge_cues(*cue_groups: Sequence[str]) -> tuple[str, ...]:
    merged: list[str] = []
    for group in cue_groups:
        for cue in group:
            text = _text(cue)
            if text and text not in merged:
                merged.append(text)
    return tuple(merged)


@dataclass(frozen=True)
class SharedInnerField:
    """共有内部場への最小射影先。"""

    memory_activation: float = 0.0
    affective_charge: float = 0.0
    relational_pull: float = 0.0
    guardedness: float = 0.0
    reopening_pull: float = 0.0
    boundary_tension: float = 0.0
    residual_tension: float = 0.0
    cues: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FieldDelta:
    """各 kernel / operator が共有内部場へ与える局所変形。"""

    source: str = ""
    memory_activation: float = 0.0
    affective_charge: float = 0.0
    relational_pull: float = 0.0
    guardedness: float = 0.0
    reopening_pull: float = 0.0
    boundary_tension: float = 0.0
    residual_tension: float = 0.0
    confidence: float = 0.0
    cues: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GreenKernelComposition:
    """Green 的射影を共有内部場へ統合した結果。"""

    field: SharedInnerField
    memory_delta: FieldDelta
    affective_delta: FieldDelta
    relational_delta: FieldDelta
    boundary_delta: FieldDelta
    residual_delta: FieldDelta

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field.to_dict(),
            "memory_delta": self.memory_delta.to_dict(),
            "affective_delta": self.affective_delta.to_dict(),
            "relational_delta": self.relational_delta.to_dict(),
            "boundary_delta": self.boundary_delta.to_dict(),
            "residual_delta": self.residual_delta.to_dict(),
        }


def project_memory_green_delta(
    *,
    temporal_membrane_bias: Mapping[str, Any] | None = None,
    memory_evidence_bundle: Mapping[str, Any] | None = None,
    autobiographical_thread: Mapping[str, Any] | None = None,
) -> FieldDelta:
    temporal = dict(temporal_membrane_bias or {})
    evidence = dict(memory_evidence_bundle or {})
    autobiographical = dict(autobiographical_thread or {})

    current_count = _list_size(evidence.get("facts_current"))
    timeline_count = _list_size(evidence.get("timeline_events"))
    source_count = _list_size(evidence.get("source_refs"))
    evidence_density = _float01((current_count + timeline_count + source_count) / 9.0)

    timeline_coherence = _float01(temporal.get("timeline_coherence"))
    reentry_pull = _float01(temporal.get("reentry_pull"))
    relation_reentry_pull = _float01(temporal.get("relation_reentry_pull"))
    continuity_pressure = _float01(temporal.get("continuity_pressure"))
    thread_strength = _float01(autobiographical.get("strength"))

    memory_activation = _float01(
        timeline_coherence * 0.28
        + continuity_pressure * 0.24
        + reentry_pull * 0.18
        + relation_reentry_pull * 0.1
        + evidence_density * 0.12
        + thread_strength * 0.08
    )
    reopening_field = _float01(
        reentry_pull * 0.52
        + relation_reentry_pull * 0.24
        + continuity_pressure * 0.12
        + thread_strength * 0.12
    )
    confidence = _float01(
        0.2
        + evidence_density * 0.36
        + timeline_coherence * 0.24
        + thread_strength * 0.2
    )
    cues = _merge_cues(
        temporal.get("cues") or (),
        autobiographical.get("reasons") or (),
        ("green_memory_projection",),
    )
    return FieldDelta(
        source="memory_green",
        memory_activation=round(memory_activation, 4),
        reopening_pull=round(reopening_field, 4),
        confidence=round(confidence, 4),
        cues=cues,
    )


def project_affective_green_delta(
    *,
    affect_blend_state: Mapping[str, Any] | None = None,
    temporal_membrane_bias: Mapping[str, Any] | None = None,
    contact_reflection_state: Mapping[str, Any] | None = None,
) -> FieldDelta:
    blend = dict(affect_blend_state or {})
    temporal = dict(temporal_membrane_bias or {})
    contact = dict(contact_reflection_state or {})

    affective_charge = _float01(
        _float01(blend.get("care")) * 0.14
        + _float01(blend.get("future_pull")) * 0.14
        + _float01(blend.get("shared_world_pull")) * 0.12
        + _float01(blend.get("distress")) * 0.2
        + _float01(blend.get("conflict_level")) * 0.2
        + _float01(blend.get("residual_tension")) * 0.12
        + _float01(temporal.get("supersession_pressure")) * 0.08
    )
    guardedness = _float01(
        _float01(blend.get("defense")) * 0.48
        + _float01(contact.get("absorb_share")) * 0.2
        + _float01(contact.get("block_share")) * 0.22
        + _float01(temporal.get("supersession_pressure")) * 0.1
    )
    boundary_tension = _float01(
        _float01(blend.get("reportability_pressure")) * 0.46
        + _float01(blend.get("residual_tension")) * 0.32
        + _float01(contact.get("reflect_share")) * 0.06
        + _float01(contact.get("absorb_share")) * 0.08
        + _float01(contact.get("block_share")) * 0.08
    )
    confidence = _float01(blend.get("confidence"))
    cues = _merge_cues(
        blend.get("cues") or (),
        temporal.get("cues") or (),
        contact.get("cues") or (),
        ("green_affective_projection",),
    )
    return FieldDelta(
        source="affective_green",
        affective_charge=round(affective_charge, 4),
        guardedness=round(guardedness, 4),
        boundary_tension=round(boundary_tension, 4),
        confidence=round(confidence, 4),
        cues=cues,
    )


def project_relational_green_delta(
    *,
    recent_dialogue_state: Mapping[str, Any] | None = None,
    discussion_thread_state: Mapping[str, Any] | None = None,
    issue_state: Mapping[str, Any] | None = None,
    autobiographical_thread: Mapping[str, Any] | None = None,
) -> FieldDelta:
    recent = dict(recent_dialogue_state or {})
    discussion = dict(discussion_thread_state or {})
    issue = dict(issue_state or {})
    autobiographical = dict(autobiographical_thread or {})

    relational_pull = _float01(
        _float01(recent.get("thread_carry")) * 0.22
        + _float01(discussion.get("thread_visibility")) * 0.24
        + _float01(discussion.get("revisit_readiness")) * 0.18
        + _float01(issue.get("pause_readiness")) * 0.1
        + _float01(issue.get("resolution_readiness")) * 0.08
        + _float01(autobiographical.get("strength")) * 0.18
    )
    reopening_pull = _float01(
        _float01(recent.get("reopen_pressure")) * 0.36
        + _float01(discussion.get("revisit_readiness")) * 0.26
        + _float01(issue.get("pause_readiness")) * 0.1
        + _float01(autobiographical.get("strength")) * 0.28
    )
    guardedness = _float01(
        _float01(issue.get("question_pressure")) * 0.42
        + max(0.0, 1.0 - _float01(discussion.get("thread_visibility"))) * 0.16
        + max(0.0, 1.0 - _float01(recent.get("thread_carry"))) * 0.12
    )
    confidence = _float01(
        0.18
        + _float01(recent.get("thread_carry")) * 0.24
        + _float01(discussion.get("thread_visibility")) * 0.28
        + _float01(autobiographical.get("strength")) * 0.2
    )
    cues = _merge_cues(
        recent.get("dominant_inputs") or (),
        discussion.get("dominant_inputs") or (),
        issue.get("dominant_inputs") or (),
        autobiographical.get("reasons") or (),
        ("green_relational_projection",),
    )
    return FieldDelta(
        source="relational_green",
        relational_pull=round(relational_pull, 4),
        guardedness=round(guardedness, 4),
        reopening_pull=round(reopening_pull, 4),
        confidence=round(confidence, 4),
        cues=cues,
    )


def project_boundary_transform_delta(
    *,
    boundary_transform: Mapping[str, Any] | None = None,
    contact_reflection_state: Mapping[str, Any] | None = None,
) -> FieldDelta:
    boundary = dict(boundary_transform or {})
    contact = dict(contact_reflection_state or {})
    boundary_tension = _float01(
        _float01(boundary.get("residual_pressure")) * 0.54
        + _float01(contact.get("block_share")) * 0.26
        + _float01(contact.get("absorb_share")) * 0.2
    )
    guardedness = _float01(
        (_text(boundary.get("gate_mode")) != "open") * 0.28
        + _float01(contact.get("block_share")) * 0.32
        + _float01(contact.get("absorb_share")) * 0.16
        + _float01(boundary.get("residual_pressure")) * 0.24
    )
    cues = _merge_cues(
        boundary.get("cues") or (),
        contact.get("cues") or (),
        ("boundary_operator_projection",),
    )
    return FieldDelta(
        source="boundary_operator",
        guardedness=round(guardedness, 4),
        boundary_tension=round(boundary_tension, 4),
        confidence=round(_float01(boundary.get("residual_pressure")), 4),
        cues=cues,
    )


def project_residual_carry_delta(
    *,
    residual_reflection: Mapping[str, Any] | None = None,
    autobiographical_thread: Mapping[str, Any] | None = None,
) -> FieldDelta:
    residual = dict(residual_reflection or {})
    autobiographical = dict(autobiographical_thread or {})
    residual_tension = _float01(
        _float01(residual.get("strength")) * 0.68
        + _float01(autobiographical.get("strength")) * 0.32
    )
    reopening_pull = _float01(
        _float01(autobiographical.get("strength")) * 0.46
        + (_text(residual.get("mode")) in {"withheld", "held_open"}) * 0.16
        + (_text(autobiographical.get("mode")) in {"unfinished_thread", "relational_lingering_thread"}) * 0.18
    )
    cues = _merge_cues(
        residual.get("cues") or (),
        autobiographical.get("reasons") or (),
        ("residual_carry_projection",),
    )
    return FieldDelta(
        source="residual_operator",
        reopening_pull=round(reopening_pull, 4),
        residual_tension=round(residual_tension, 4),
        confidence=round(_float01(residual.get("strength")), 4),
        cues=cues,
    )


def compose_shared_inner_field(
    *,
    memory_delta: FieldDelta,
    affective_delta: FieldDelta,
    relational_delta: FieldDelta,
    boundary_delta: FieldDelta,
    residual_delta: FieldDelta,
) -> GreenKernelComposition:
    field = SharedInnerField(
        memory_activation=round(
            _float01(
                memory_delta.memory_activation
                + boundary_delta.memory_activation
                + residual_delta.memory_activation
            ),
            4,
        ),
        affective_charge=round(
            _float01(
                affective_delta.affective_charge
                + memory_delta.affective_charge
                + residual_delta.affective_charge
            ),
            4,
        ),
        relational_pull=round(
            _float01(
                relational_delta.relational_pull
                + memory_delta.relational_pull
                + residual_delta.relational_pull
            ),
            4,
        ),
        guardedness=round(
            _float01(
                affective_delta.guardedness
                + relational_delta.guardedness
                + boundary_delta.guardedness
            ),
            4,
        ),
        reopening_pull=round(
            _float01(
                memory_delta.reopening_pull
                + relational_delta.reopening_pull
                + residual_delta.reopening_pull
            ),
            4,
        ),
        boundary_tension=round(
            _float01(
                affective_delta.boundary_tension
                + boundary_delta.boundary_tension
            ),
            4,
        ),
        residual_tension=round(
            _float01(
                residual_delta.residual_tension
                + affective_delta.residual_tension
                + boundary_delta.residual_tension
            ),
            4,
        ),
        cues=_merge_cues(
            memory_delta.cues,
            affective_delta.cues,
            relational_delta.cues,
            boundary_delta.cues,
            residual_delta.cues,
        ),
    )
    return GreenKernelComposition(
        field=field,
        memory_delta=memory_delta,
        affective_delta=affective_delta,
        relational_delta=relational_delta,
        boundary_delta=boundary_delta,
        residual_delta=residual_delta,
    )


def build_green_kernel_composition(
    *,
    temporal_membrane_bias: Mapping[str, Any] | None = None,
    memory_evidence_bundle: Mapping[str, Any] | None = None,
    affect_blend_state: Mapping[str, Any] | None = None,
    recent_dialogue_state: Mapping[str, Any] | None = None,
    discussion_thread_state: Mapping[str, Any] | None = None,
    issue_state: Mapping[str, Any] | None = None,
    contact_reflection_state: Mapping[str, Any] | None = None,
    boundary_transform: Mapping[str, Any] | None = None,
    residual_reflection: Mapping[str, Any] | None = None,
    autobiographical_thread: Mapping[str, Any] | None = None,
) -> GreenKernelComposition:
    memory_delta = project_memory_green_delta(
        temporal_membrane_bias=temporal_membrane_bias,
        memory_evidence_bundle=memory_evidence_bundle,
        autobiographical_thread=autobiographical_thread,
    )
    affective_delta = project_affective_green_delta(
        affect_blend_state=affect_blend_state,
        temporal_membrane_bias=temporal_membrane_bias,
        contact_reflection_state=contact_reflection_state,
    )
    relational_delta = project_relational_green_delta(
        recent_dialogue_state=recent_dialogue_state,
        discussion_thread_state=discussion_thread_state,
        issue_state=issue_state,
        autobiographical_thread=autobiographical_thread,
    )
    boundary_delta = project_boundary_transform_delta(
        boundary_transform=boundary_transform,
        contact_reflection_state=contact_reflection_state,
    )
    residual_delta = project_residual_carry_delta(
        residual_reflection=residual_reflection,
        autobiographical_thread=autobiographical_thread,
    )
    return compose_shared_inner_field(
        memory_delta=memory_delta,
        affective_delta=affective_delta,
        relational_delta=relational_delta,
        boundary_delta=boundary_delta,
        residual_delta=residual_delta,
    )
