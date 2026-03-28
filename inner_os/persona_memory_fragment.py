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


@dataclass(frozen=True)
class PersonaMemoryFragment:
    """状況に応じて想起される人格断片。"""

    fragment_id: str
    kind: str
    summary: str
    anchor: str = ""
    intensity: float = 0.0
    boundary_mode: str = ""
    related_person_id: str = ""
    group_thread_id: str = ""
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "kind": self.kind,
            "summary": self.summary,
            "anchor": self.anchor,
            "intensity": round(float(self.intensity), 4),
            "boundary_mode": self.boundary_mode,
            "related_person_id": self.related_person_id,
            "group_thread_id": self.group_thread_id,
            "tags": list(self.tags),
        }


def build_persona_memory_fragments(
    *,
    self_state: Mapping[str, Any],
    relation_bias_strength: float,
    related_person_ids: Sequence[str],
    social_topology_state: Mapping[str, Any],
    relational_style_memory_state: Mapping[str, Any],
    cultural_conversation_state: Mapping[str, Any],
    protection_mode: Mapping[str, Any],
    grice_guard_state: Mapping[str, Any],
) -> list[PersonaMemoryFragment]:
    state = dict(self_state or {})
    topology = dict(social_topology_state or {})
    relation_style = dict(relational_style_memory_state or {})
    cultural = dict(cultural_conversation_state or {})
    protection = dict(protection_mode or {})
    grice = dict(grice_guard_state or {})

    fragments: list[PersonaMemoryFragment] = []

    identity_summary = _text(state.get("identity_arc_summary"))
    identity_kind = _text(state.get("identity_arc_kind"))
    identity_anchor = _text(state.get("memory_anchor"))
    identity_related_person = _text(state.get("related_person_id"))
    identity_group_thread = _text(state.get("group_thread_focus"))
    identity_stability = _float01(state.get("identity_arc_stability"))
    if identity_summary or identity_kind:
        fragments.append(
            PersonaMemoryFragment(
                fragment_id="identity_arc",
                kind="identity",
                summary=identity_summary or identity_kind,
                anchor=identity_anchor,
                intensity=max(identity_stability, _float01(state.get("long_term_theme_strength"))),
                related_person_id=identity_related_person,
                group_thread_id=identity_group_thread,
                tags=tuple(
                    item
                    for item in (
                        identity_kind,
                        _text(state.get("identity_arc_phase")),
                        _text(state.get("identity_arc_open_tension")),
                    )
                    if item
                ),
            )
        )

    long_term_theme_summary = _text(state.get("long_term_theme_summary"))
    long_term_theme_focus = _text(state.get("long_term_theme_focus"))
    if long_term_theme_summary or long_term_theme_focus:
        fragments.append(
            PersonaMemoryFragment(
                fragment_id="long_term_theme",
                kind="theme",
                summary=long_term_theme_summary or long_term_theme_focus,
                anchor=_text(state.get("long_term_theme_anchor")),
                intensity=_float01(state.get("long_term_theme_strength")),
                tags=tuple(
                    item
                    for item in (
                        _text(state.get("long_term_theme_kind")),
                        long_term_theme_focus,
                    )
                    if item
                ),
            )
        )

    relation_seed_summary = _text(state.get("relation_seed_summary"))
    related_person_id = _text(state.get("related_person_id")) or (str(related_person_ids[0]).strip() if related_person_ids else "")
    if relation_seed_summary or related_person_id:
        fragments.append(
            PersonaMemoryFragment(
                fragment_id="relation_seed",
                kind="relation",
                summary=relation_seed_summary or f"relation thread with {related_person_id}",
                anchor=_text(state.get("partner_address_hint")) or _text(state.get("partner_social_interpretation")),
                intensity=max(_float01(state.get("relation_seed_strength")), _float01(relation_bias_strength)),
                related_person_id=related_person_id,
                tags=tuple(
                    item
                    for item in (
                        _text(state.get("partner_timing_hint")),
                        _text(state.get("partner_stance_hint")),
                        _text(state.get("partner_social_interpretation")),
                    )
                    if item
                ),
            )
        )

    style_state = _text(relation_style.get("state"))
    style_summary = " / ".join(
        item
        for item in (
            style_state,
            _text(relation_style.get("banter_style")),
            _text(cultural.get("state")),
        )
        if item
    )
    style_intensity = max(
        _float01(relation_style.get("playful_ceiling")),
        _float01(relation_style.get("advice_tolerance")),
        _float01(cultural.get("directness_ceiling")),
    )
    if style_summary:
        fragments.append(
            PersonaMemoryFragment(
                fragment_id="style_memory",
                kind="style",
                summary=style_summary,
                intensity=style_intensity,
                tags=tuple(
                    item
                    for item in (
                        _text(relation_style.get("banter_style")),
                        _text(cultural.get("state")),
                        _text(topology.get("state")),
                    )
                    if item
                ),
            )
        )

    boundary_summary = " / ".join(
        item
        for item in (
            _text(protection.get("mode")),
            _text(grice.get("state")),
            _text(topology.get("state")),
        )
        if item
    )
    boundary_intensity = max(
        _float01(protection.get("strength")),
        _float01(grice.get("knownness_pressure")),
        _float01(topology.get("visibility_pressure")),
    )
    if boundary_summary:
        fragments.append(
            PersonaMemoryFragment(
                fragment_id="boundary_memory",
                kind="boundary",
                summary=boundary_summary,
                boundary_mode=_text(topology.get("state")) or _text(protection.get("mode")),
                intensity=boundary_intensity,
                related_person_id=related_person_id,
                group_thread_id=identity_group_thread,
                tags=tuple(
                    item
                    for item in (
                        _text(grice.get("state")),
                        _text(protection.get("mode")),
                        _text(topology.get("state")),
                    )
                    if item
                ),
            )
        )

    return [fragment for fragment in fragments if fragment.summary]
