from __future__ import annotations

from ..self_model.models import SelfState
from ..world_model.models import WorldState
from .models import Affordance, ObservationBundle


def _float01(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, numeric))


def _partner_affordance_tags(
    entity_id: str,
    *,
    observation: ObservationBundle,
    partner_context: dict[str, dict[str, object]] | None,
    self_state: SelfState,
) -> tuple[dict[str, str], float]:
    if not partner_context:
        return {}, 0.0
    entity = next((item for item in observation.entities if item.entity_id == entity_id), None)
    if entity is None:
        return {}, 0.0
    person_id = (entity.attributes.get("person_id_hint") or "").strip()
    profile = partner_context.get(person_id) if person_id else None
    if not isinstance(profile, dict):
        return {}, 0.0

    affiliation_bias = _float01(profile.get("affiliation_bias"), 0.45)
    caution_bias = _float01(profile.get("caution_bias"), 0.4)
    community_bias = _float01(profile.get("community_bias"), 0.5)
    culture_bias = _float01(profile.get("culture_bias"), 0.5)
    stable_traits = profile.get("stable_traits") if isinstance(profile.get("stable_traits"), dict) else {}
    community_marker = _float01(stable_traits.get("community_marker"), 0.0)
    culture_marker = _float01(stable_traits.get("culture_marker"), 0.0)
    role_marker = _float01(stable_traits.get("role_marker"), 0.0)

    community_resonance = min(
        1.0,
        community_marker * community_bias * 0.58
        + culture_marker * culture_bias * 0.42
        + role_marker * (0.14 + affiliation_bias * 0.18),
    )
    timing_gate = min(
        1.0,
        max(
            0.15,
            0.44
            + affiliation_bias * 0.26
            - caution_bias * 0.2
            - _float01(self_state.social_tension, 0.0) * 0.14,
        ),
    )
    if timing_gate < 0.42:
        stance_hint = "guarded"
        timing_hint = "delayed"
    elif community_resonance > 0.45 and affiliation_bias > caution_bias:
        stance_hint = "familiar"
        timing_hint = "open"
    else:
        stance_hint = "respectful"
        timing_hint = "measured"

    address_hint = "companion" if community_resonance > 0.48 and affiliation_bias > 0.6 else (
        "respectful" if caution_bias > 0.62 or timing_gate < 0.42 else "neutral"
    )
    return (
        {
            "partner_person_id": person_id,
            "stance_hint": stance_hint,
            "timing_hint": timing_hint,
            "address_hint": address_hint,
        },
        min(1.0, community_resonance * 0.46 + timing_gate * 0.34),
    )


def infer_affordances(
    observation: ObservationBundle,
    world_state: WorldState,
    self_state: SelfState,
    partner_context: dict[str, dict[str, object]] | None = None,
) -> dict[str, list[Affordance]]:
    """観測対象に対して最小の行為可能性を返す。"""
    affordances: dict[str, list[Affordance]] = {}
    for entity in observation.entities:
        score = max(0.1, 1.0 - observation.observation_uncertainty * 0.5)
        context_tags, social_lift = _partner_affordance_tags(
            entity.entity_id,
            observation=observation,
            partner_context=partner_context,
            self_state=self_state,
        )
        entity_affordances = [
            Affordance(
                action="inspect",
                score=round(min(1.0, score + social_lift * 0.08), 4),
                constraints=["needs_more_evidence"] if observation.observation_uncertainty > 0.5 else [],
                risk=round(self_state.safety_margin * 0.1, 4),
                context_tags=context_tags,
            )
        ]
        if context_tags:
            engage_constraints: list[str] = []
            if context_tags.get("timing_hint") == "delayed":
                engage_constraints.append("wait_for_social_timing")
            entity_affordances.append(
                Affordance(
                    action="engage",
                    score=round(min(1.0, score * 0.72 + social_lift * 0.24), 4),
                    constraints=engage_constraints,
                    risk=round(self_state.safety_margin * 0.08 + _float01(self_state.social_tension, 0.0) * 0.06, 4),
                    context_tags=context_tags,
                )
            )
        affordances[entity.entity_id] = entity_affordances
    return affordances
