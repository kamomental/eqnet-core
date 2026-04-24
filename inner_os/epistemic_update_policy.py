from __future__ import annotations

from typing import Any, Mapping

from .epistemic_state import EpistemicState, coerce_epistemic_state


EPISTEMIC_UPDATE_WEIGHTS = {
    "freshness_recency": 0.54,
    "freshness_direct": 0.28,
    "freshness_observation": 0.18,
    "confidence_direct": 0.46,
    "confidence_kind": 0.34,
    "confidence_observation": 0.2,
    "stale_inverse_freshness": 0.42,
    "stale_change": 0.34,
    "stale_low_confidence": 0.24,
    "verification_stale": 0.44,
    "verification_change": 0.28,
    "verification_low_confidence": 0.18,
    "verification_dynamic_domain": 0.1,
    "caution_verification": 0.54,
    "caution_low_confidence": 0.18,
    "caution_inverse_freshness": 0.14,
    "caution_change": 0.14,
    "blend_floor": 0.32,
    "blend_base": 0.24,
    "blend_carry_bias": 0.34,
    "blend_revision_bias": 0.3,
    "blend_directive_bias": 0.16,
}


class EpistemicUpdatePolicy:
    """鮮度・出典・変化可能性から知識姿勢を更新する。"""

    def evolve(
        self,
        *,
        previous_epistemic: Mapping[str, Any] | EpistemicState | None = None,
        current_state: Mapping[str, Any] | None = None,
        memory_evidence: Mapping[str, Any] | None = None,
        observation_state: Mapping[str, Any] | None = None,
    ) -> EpistemicState:
        previous = coerce_epistemic_state(previous_epistemic)
        current = dict(current_state or {})
        evidence = dict(memory_evidence or {})
        observation = dict(observation_state or {})

        freshness_target = _clamp01(
            _freshness_from_days(
                current.get("freshness_days", evidence.get("freshness_days"))
            ) * EPISTEMIC_UPDATE_WEIGHTS["freshness_recency"]
            + _float01(
                evidence.get("freshness", current.get("freshness")),
                previous.freshness,
            ) * EPISTEMIC_UPDATE_WEIGHTS["freshness_direct"]
            + _float01(
                observation.get("freshness", observation.get("observation_freshness")),
                0.0,
            ) * EPISTEMIC_UPDATE_WEIGHTS["freshness_observation"]
        )
        source_confidence_target = _clamp01(
            _float01(
                evidence.get("source_confidence", current.get("source_confidence")),
                previous.source_confidence,
            ) * EPISTEMIC_UPDATE_WEIGHTS["confidence_direct"]
            + _kind_confidence_bias(
                str(evidence.get("record_kind") or current.get("record_kind") or "")
            ) * EPISTEMIC_UPDATE_WEIGHTS["confidence_kind"]
            + _float01(
                observation.get("source_confidence", observation.get("confidence")),
                0.0,
            ) * EPISTEMIC_UPDATE_WEIGHTS["confidence_observation"]
        )
        change_likelihood_target = _clamp01(
            max(
                _float01(current.get("change_likelihood"), 0.0),
                _float01(evidence.get("change_likelihood"), 0.0),
                _float01(observation.get("change_likelihood"), 0.0),
                _domain_change_bias(
                    str(
                        current.get("knowledge_scope")
                        or evidence.get("knowledge_scope")
                        or current.get("source_scope")
                        or ""
                    )
                ),
            )
        )
        stale_risk_target = _clamp01(
            (1.0 - freshness_target) * EPISTEMIC_UPDATE_WEIGHTS["stale_inverse_freshness"]
            + change_likelihood_target * EPISTEMIC_UPDATE_WEIGHTS["stale_change"]
            + (1.0 - source_confidence_target) * EPISTEMIC_UPDATE_WEIGHTS["stale_low_confidence"]
        )
        dynamic_domain_bias = _domain_change_bias(
            str(
                current.get("source_type")
                or evidence.get("source_type")
                or current.get("knowledge_scope")
                or evidence.get("knowledge_scope")
                or ""
            )
        )
        verification_pressure_target = _clamp01(
            stale_risk_target * EPISTEMIC_UPDATE_WEIGHTS["verification_stale"]
            + change_likelihood_target * EPISTEMIC_UPDATE_WEIGHTS["verification_change"]
            + (1.0 - source_confidence_target) * EPISTEMIC_UPDATE_WEIGHTS["verification_low_confidence"]
            + dynamic_domain_bias * EPISTEMIC_UPDATE_WEIGHTS["verification_dynamic_domain"]
        )
        epistemic_caution_target = _clamp01(
            verification_pressure_target * EPISTEMIC_UPDATE_WEIGHTS["caution_verification"]
            + (1.0 - source_confidence_target) * EPISTEMIC_UPDATE_WEIGHTS["caution_low_confidence"]
            + (1.0 - freshness_target) * EPISTEMIC_UPDATE_WEIGHTS["caution_inverse_freshness"]
            + change_likelihood_target * EPISTEMIC_UPDATE_WEIGHTS["caution_change"]
        )

        carry_signal = _clamp01(freshness_target * 0.5 + source_confidence_target * 0.5)
        revision_signal = _clamp01(stale_risk_target * 0.5 + change_likelihood_target * 0.5)
        adaptive_update_strength = _clamp01(
            EPISTEMIC_UPDATE_WEIGHTS["blend_base"]
            + carry_signal * EPISTEMIC_UPDATE_WEIGHTS["blend_carry_bias"]
            + revision_signal * EPISTEMIC_UPDATE_WEIGHTS["blend_revision_bias"]
            + verification_pressure_target * EPISTEMIC_UPDATE_WEIGHTS["blend_directive_bias"]
        )
        update_strength = max(
            EPISTEMIC_UPDATE_WEIGHTS["blend_floor"],
            adaptive_update_strength,
            _float01(current.get("epistemic_update_strength"), 0.42),
        )

        freshness = _blend(previous.freshness, freshness_target, update_strength)
        source_confidence = _blend(previous.source_confidence, source_confidence_target, update_strength)
        verification_pressure = _blend(previous.verification_pressure, verification_pressure_target, update_strength)
        change_likelihood = _blend(previous.change_likelihood, change_likelihood_target, update_strength)
        stale_risk = _blend(previous.stale_risk, stale_risk_target, update_strength)
        epistemic_caution = _blend(previous.epistemic_caution, epistemic_caution_target, update_strength)

        return EpistemicState(
            freshness=freshness,
            source_confidence=source_confidence,
            verification_pressure=verification_pressure,
            change_likelihood=change_likelihood,
            stale_risk=stale_risk,
            epistemic_caution=epistemic_caution,
            dominant_posture=_dominant_posture(
                freshness=freshness,
                source_confidence=source_confidence,
                verification_pressure=verification_pressure,
                change_likelihood=change_likelihood,
                stale_risk=stale_risk,
                epistemic_caution=epistemic_caution,
            ),
        )


def derive_epistemic_state(
    *,
    previous_epistemic: Mapping[str, Any] | EpistemicState | None = None,
    current_state: Mapping[str, Any] | None = None,
    memory_evidence: Mapping[str, Any] | None = None,
    observation_state: Mapping[str, Any] | None = None,
) -> EpistemicState:
    return EpistemicUpdatePolicy().evolve(
        previous_epistemic=previous_epistemic,
        current_state=current_state,
        memory_evidence=memory_evidence,
        observation_state=observation_state,
    )


def _freshness_from_days(value: Any) -> float:
    try:
        days = max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0
    return _clamp01(1.0 - min(days, 30.0) / 30.0)


def _kind_confidence_bias(record_kind: str) -> float:
    normalized = str(record_kind or "").strip()
    return {
        "verified": 0.88,
        "observed_real": 0.76,
        "transferred_learning": 0.66,
        "experienced_sim": 0.56,
        "reconstructed": 0.38,
    }.get(normalized, 0.5)


def _domain_change_bias(source_scope: str) -> float:
    normalized = str(source_scope or "").strip()
    if normalized in {
        "live_trend",
        "news",
        "social",
        "slang",
        "product",
        "law",
        "people",
        "community_live",
    }:
        return 0.78
    if normalized in {"culture", "community", "relationship", "shared_thread"}:
        return 0.46
    if normalized in {"history", "place", "autobiographical"}:
        return 0.22
    return 0.32


def _dominant_posture(
    *,
    freshness: float,
    source_confidence: float,
    verification_pressure: float,
    change_likelihood: float,
    stale_risk: float,
    epistemic_caution: float,
) -> str:
    if verification_pressure >= 0.62:
        return "reverify"
    if change_likelihood >= 0.62 and freshness <= 0.48:
        return "update_priority"
    if epistemic_caution >= 0.56 and source_confidence <= 0.46:
        return "hold_ambiguity"
    if freshness >= 0.64 and source_confidence >= 0.62 and stale_risk <= 0.4:
        return "carry_forward"
    return "carry_forward"


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
