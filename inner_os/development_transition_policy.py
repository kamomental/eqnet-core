from __future__ import annotations

from typing import Any, Mapping

from .growth_state import GrowthState, coerce_growth_state


DEVELOPMENT_TRANSITION_WEIGHTS = {
    "relational_belonging": 0.24,
    "relational_trust": 0.34,
    "relational_identity": 0.12,
    "relational_continuity": 0.14,
    "relational_monument": 0.08,
    "relational_social_grounding": 0.08,
    "epistemic_abstraction": 0.24,
    "epistemic_reconsolidation": 0.2,
    "epistemic_identity": 0.12,
    "epistemic_forgetting_relief": 0.22,
    "epistemic_norm": 0.1,
    "epistemic_association": 0.12,
    "expressive_social_update": 0.18,
    "expressive_carry": 0.24,
    "expressive_lexical": 0.14,
    "expressive_forgetting_relief": 0.14,
    "expressive_lightness": 0.16,
    "expressive_relation": 0.14,
    "residue_replay": 0.24,
    "residue_autobio": 0.24,
    "residue_monument": 0.16,
    "residue_association": 0.14,
    "residue_thread": 0.12,
    "residue_identity": 0.1,
    "playfulness_lightness": 0.28,
    "playfulness_lexical": 0.16,
    "playfulness_trust": 0.16,
    "playfulness_norm_relief": 0.14,
    "playfulness_expressive": 0.14,
    "playfulness_forgetting_relief": 0.12,
    "coherence_identity": 0.28,
    "coherence_continuity": 0.24,
    "coherence_role": 0.14,
    "coherence_epistemic": 0.1,
    "coherence_forgetting_relief": 0.12,
    "coherence_residue": 0.12,
    "social_blend_floor": 0.32,
    "identity_blend_floor": 0.28,
}


class DevelopmentTransitionPolicy:
    """既存 slow-state 断片を GrowthState に束ねる小さな更新則。"""

    def evolve(
        self,
        *,
        previous_growth: Mapping[str, Any] | GrowthState | None = None,
        development_state: Mapping[str, Any] | None = None,
        forgetting_snapshot: Mapping[str, Any] | None = None,
        sleep_consolidation: Mapping[str, Any] | None = None,
        transfer_package: Mapping[str, Any] | None = None,
    ) -> GrowthState:
        previous = coerce_growth_state(previous_growth)
        development = dict(development_state or {})
        forgetting = dict(forgetting_snapshot or {})
        sleep = dict(sleep_consolidation or {})
        carry = _extract_transfer_metrics(transfer_package)

        belonging = _float01(development.get("belonging"))
        trust_bias = _float01(development.get("trust_bias"))
        norm_pressure = _float01(development.get("norm_pressure"))
        role_commitment = _float01(development.get("role_commitment"))
        social_update_strength = _float01(development.get("social_update_strength"), previous.social_update_strength)
        identity_update_strength = _float01(development.get("identity_update_strength"), previous.identity_update_strength)

        forgetting_pressure = _float01(forgetting.get("forgetting_pressure"))
        forgetting_relief = _clamp01(1.0 - forgetting_pressure)

        replay_priority = _float01(sleep.get("replay_priority"))
        reconsolidation_priority = _float01(sleep.get("reconsolidation_priority"))
        autobiographical_pull = _float01(sleep.get("autobiographical_pull"))
        abstraction_readiness = _float01(sleep.get("abstraction_readiness"))
        identity_preservation_bias = _float01(sleep.get("identity_preservation_bias"))
        expressive_style_carry_bias = _float01(sleep.get("expressive_style_carry_bias"))

        relational_target = _clamp01(
            belonging * DEVELOPMENT_TRANSITION_WEIGHTS["relational_belonging"]
            + trust_bias * DEVELOPMENT_TRANSITION_WEIGHTS["relational_trust"]
            + identity_preservation_bias * DEVELOPMENT_TRANSITION_WEIGHTS["relational_identity"]
            + carry["continuity_score"] * DEVELOPMENT_TRANSITION_WEIGHTS["relational_continuity"]
            + carry["monument_salience"] * DEVELOPMENT_TRANSITION_WEIGHTS["relational_monument"]
            + carry["social_grounding"] * DEVELOPMENT_TRANSITION_WEIGHTS["relational_social_grounding"]
        )
        epistemic_target = _clamp01(
            abstraction_readiness * DEVELOPMENT_TRANSITION_WEIGHTS["epistemic_abstraction"]
            + reconsolidation_priority * DEVELOPMENT_TRANSITION_WEIGHTS["epistemic_reconsolidation"]
            + identity_preservation_bias * DEVELOPMENT_TRANSITION_WEIGHTS["epistemic_identity"]
            + forgetting_relief * DEVELOPMENT_TRANSITION_WEIGHTS["epistemic_forgetting_relief"]
            + norm_pressure * DEVELOPMENT_TRANSITION_WEIGHTS["epistemic_norm"]
            + carry["association_reweighting_bias"] * DEVELOPMENT_TRANSITION_WEIGHTS["epistemic_association"]
        )
        expressive_target = _clamp01(
            social_update_strength * DEVELOPMENT_TRANSITION_WEIGHTS["expressive_social_update"]
            + max(expressive_style_carry_bias, carry["expressive_style_carry_bias"]) * DEVELOPMENT_TRANSITION_WEIGHTS["expressive_carry"]
            + carry["lexical_variation_bias"] * DEVELOPMENT_TRANSITION_WEIGHTS["expressive_lexical"]
            + forgetting_relief * DEVELOPMENT_TRANSITION_WEIGHTS["expressive_forgetting_relief"]
            + carry["lightness_room"] * DEVELOPMENT_TRANSITION_WEIGHTS["expressive_lightness"]
            + carry["relational_continuity_carry_bias"] * DEVELOPMENT_TRANSITION_WEIGHTS["expressive_relation"]
        )
        residue_target = _clamp01(
            replay_priority * DEVELOPMENT_TRANSITION_WEIGHTS["residue_replay"]
            + autobiographical_pull * DEVELOPMENT_TRANSITION_WEIGHTS["residue_autobio"]
            + carry["monument_salience"] * DEVELOPMENT_TRANSITION_WEIGHTS["residue_monument"]
            + carry["association_reweighting_bias"] * DEVELOPMENT_TRANSITION_WEIGHTS["residue_association"]
            + carry["autobiographical_thread_strength"] * DEVELOPMENT_TRANSITION_WEIGHTS["residue_thread"]
            + identity_update_strength * DEVELOPMENT_TRANSITION_WEIGHTS["residue_identity"]
        )
        playfulness_target = _clamp01(
            carry["lightness_room"] * DEVELOPMENT_TRANSITION_WEIGHTS["playfulness_lightness"]
            + carry["lexical_variation_bias"] * DEVELOPMENT_TRANSITION_WEIGHTS["playfulness_lexical"]
            + trust_bias * DEVELOPMENT_TRANSITION_WEIGHTS["playfulness_trust"]
            + _clamp01(1.0 - norm_pressure) * DEVELOPMENT_TRANSITION_WEIGHTS["playfulness_norm_relief"]
            + expressive_target * DEVELOPMENT_TRANSITION_WEIGHTS["playfulness_expressive"]
            + forgetting_relief * DEVELOPMENT_TRANSITION_WEIGHTS["playfulness_forgetting_relief"]
        )
        coherence_target = _clamp01(
            identity_preservation_bias * DEVELOPMENT_TRANSITION_WEIGHTS["coherence_identity"]
            + carry["continuity_score"] * DEVELOPMENT_TRANSITION_WEIGHTS["coherence_continuity"]
            + role_commitment * DEVELOPMENT_TRANSITION_WEIGHTS["coherence_role"]
            + epistemic_target * DEVELOPMENT_TRANSITION_WEIGHTS["coherence_epistemic"]
            + forgetting_relief * DEVELOPMENT_TRANSITION_WEIGHTS["coherence_forgetting_relief"]
            + residue_target * DEVELOPMENT_TRANSITION_WEIGHTS["coherence_residue"]
        )

        social_blend = max(
            DEVELOPMENT_TRANSITION_WEIGHTS["social_blend_floor"],
            _clamp01(social_update_strength * 0.74 + carry["reopening_relief"] * 0.18),
        )
        identity_blend = max(
            DEVELOPMENT_TRANSITION_WEIGHTS["identity_blend_floor"],
            _clamp01(identity_update_strength * 0.72 + carry["reopening_relief"] * 0.2),
        )

        relational_trust = _blend(previous.relational_trust, relational_target, social_blend)
        epistemic_maturity = _blend(previous.epistemic_maturity, epistemic_target, identity_blend)
        expressive_range = _blend(previous.expressive_range, expressive_target, social_blend)
        residue_integration = _blend(previous.residue_integration, residue_target, identity_blend)
        playfulness_range = _blend(previous.playfulness_range, playfulness_target, social_blend)
        self_coherence = _blend(previous.self_coherence, coherence_target, identity_blend)

        axis_deltas = {
            "relational_trust": relational_trust - previous.relational_trust,
            "epistemic_maturity": epistemic_maturity - previous.epistemic_maturity,
            "expressive_range": expressive_range - previous.expressive_range,
            "residue_integration": residue_integration - previous.residue_integration,
            "playfulness_range": playfulness_range - previous.playfulness_range,
            "self_coherence": self_coherence - previous.self_coherence,
        }

        return GrowthState(
            relational_trust=relational_trust,
            epistemic_maturity=epistemic_maturity,
            expressive_range=expressive_range,
            residue_integration=residue_integration,
            playfulness_range=playfulness_range,
            self_coherence=self_coherence,
            social_update_strength=social_update_strength,
            identity_update_strength=identity_update_strength,
            dominant_transition=max(
                axis_deltas.items(),
                key=lambda item: (abs(item[1]), item[0]),
            )[0],
        )


def derive_growth_state(
    *,
    previous_growth: Mapping[str, Any] | GrowthState | None = None,
    development_state: Mapping[str, Any] | None = None,
    forgetting_snapshot: Mapping[str, Any] | None = None,
    sleep_consolidation: Mapping[str, Any] | None = None,
    transfer_package: Mapping[str, Any] | None = None,
) -> GrowthState:
    return DevelopmentTransitionPolicy().evolve(
        previous_growth=previous_growth,
        development_state=development_state,
        forgetting_snapshot=forgetting_snapshot,
        sleep_consolidation=sleep_consolidation,
        transfer_package=transfer_package,
    )


def _extract_transfer_metrics(transfer_package: Mapping[str, Any] | None) -> dict[str, float]:
    package = dict(transfer_package or {})
    portable_state = _mapping(package.get("portable_state"))
    same_turn = _mapping(portable_state.get("same_turn"))
    carry = _mapping(portable_state.get("carry"))
    runtime_seed = _mapping(package.get("runtime_seed"))
    memory_carry = _mapping(carry.get("memory_carry"))
    monument_carry = _mapping(memory_carry.get("monument_carry"))
    expressive_style_state = _mapping(same_turn.get("expressive_style_state"))
    relational_style_memory_state = _mapping(same_turn.get("relational_style_memory_state"))

    return {
        "continuity_score": max(
            _float01(runtime_seed.get("continuity_score")),
            _float01(_nested_value(carry, "daily_carry_summary", "same_turn_alignment", "same_turn_alignment")),
        ),
        "social_grounding": _float01(runtime_seed.get("social_grounding")),
        "monument_salience": max(
            _float01(runtime_seed.get("monument_salience")),
            _float01(monument_carry.get("monument_salience")),
        ),
        "association_reweighting_bias": _float01(runtime_seed.get("association_reweighting_bias")),
        "expressive_style_carry_bias": _float01(runtime_seed.get("expressive_style_carry_bias")),
        "lexical_variation_bias": max(
            _float01(runtime_seed.get("lexical_variation_bias")),
            _float01(relational_style_memory_state.get("lexical_variation_bias")),
        ),
        "lightness_room": _float01(expressive_style_state.get("lightness_room")),
        "relational_continuity_carry_bias": _float01(runtime_seed.get("relational_continuity_carry_bias")),
        "autobiographical_thread_strength": max(
            _float01(runtime_seed.get("autobiographical_thread_strength")),
            _float01(_nested_value(carry, "memory_carry", "autobiographical_thread", "strength")),
        ),
        "reopening_relief": _float01(runtime_seed.get("initiative_followup_bias")),
    }


def _nested_value(mapping: Mapping[str, Any] | None, *path: str) -> Any:
    current: Any = dict(mapping or {})
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _float01(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return _clamp01(numeric)


def _blend(previous: float, target: float, strength: float) -> float:
    alpha = _clamp01(strength)
    return _clamp01(previous * (1.0 - alpha) + target * alpha)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
