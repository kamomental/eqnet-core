from __future__ import annotations

from ..value_system.models import ValueState
from .models import Affordance, ObservationBundle, SymbolGrounding


def ground_symbols(
    tokens: list[str],
    observation: ObservationBundle,
    affordances: dict[str, list[Affordance]],
    value_state: ValueState,
    partner_context: dict[str, dict[str, object]] | None = None,
) -> dict[str, SymbolGrounding]:
    """token を percept/action/value に結びつける。"""
    grounded: dict[str, SymbolGrounding] = {}
    entity_ids = [entity.entity_id for entity in observation.entities]
    affordance_refs = [entity_id for entity_id, items in affordances.items() if items]
    action_refs = [items[0].action for items in affordances.values() if items]
    value_refs = [name for name, value in value_state.value_axes.items() if value > 0.0]

    context_tags: dict[str, str] = {}
    partner_person_id = ""
    for entity in observation.entities:
        hinted_person = (entity.attributes.get("person_id_hint") or "").strip()
        if not hinted_person:
            continue
        partner_person_id = hinted_person
        first_affordance = affordances.get(entity.entity_id, [None])[0]
        if first_affordance is not None and first_affordance.context_tags:
            context_tags = dict(first_affordance.context_tags)
        break

    for token in tokens:
        token_context = dict(context_tags)
        if partner_person_id:
            token_context["reference_person_id"] = partner_person_id
        if partner_context and partner_person_id in partner_context:
            token_context.setdefault("social_reference", "partner-aware")
        grounded[token] = SymbolGrounding(
            token=token,
            percept_refs=entity_ids[:2],
            affordance_refs=affordance_refs[:2],
            value_refs=value_refs[:2],
            action_refs=action_refs[:2],
            confidence=0.5 if entity_ids else 0.1,
            context_tags=token_context,
        )
    return grounded


def summarize_partner_grounding(
    observation: ObservationBundle,
    affordances: dict[str, list[Affordance]],
    symbol_groundings: dict[str, SymbolGrounding],
) -> dict[str, str]:
    for entity in observation.entities:
        person_id = (entity.attributes.get("person_id_hint") or "").strip()
        if not person_id:
            continue
        for affordance in affordances.get(entity.entity_id, []):
            if affordance.context_tags:
                tags = dict(affordance.context_tags)
                tags.setdefault("reference_person_id", person_id)
                return tags
        for grounding in symbol_groundings.values():
            if grounding.context_tags.get("reference_person_id") == person_id:
                return dict(grounding.context_tags)
    for grounding in symbol_groundings.values():
        if grounding.context_tags:
            return dict(grounding.context_tags)
    return {}
