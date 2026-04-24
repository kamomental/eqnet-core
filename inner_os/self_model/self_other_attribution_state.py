from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SelfOtherAttributionState:
    """自己・他者・共有の帰属を連続量として推定した状態。"""

    dominant_attribution: str = "unknown"
    self_likelihood: float = 0.0
    other_likelihood: float = 0.0
    shared_likelihood: float = 0.0
    unknown_likelihood: float = 1.0
    appearance_match: float = 0.0
    contingency_match: float = 0.0
    perspective_match: float = 0.0
    sensorimotor_consistency: float = 0.0
    attribution_confidence: float = 0.0
    ambiguity: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dominant_attribution": self.dominant_attribution,
            "self_likelihood": round(self.self_likelihood, 4),
            "other_likelihood": round(self.other_likelihood, 4),
            "shared_likelihood": round(self.shared_likelihood, 4),
            "unknown_likelihood": round(self.unknown_likelihood, 4),
            "appearance_match": round(self.appearance_match, 4),
            "contingency_match": round(self.contingency_match, 4),
            "perspective_match": round(self.perspective_match, 4),
            "sensorimotor_consistency": round(self.sensorimotor_consistency, 4),
            "attribution_confidence": round(self.attribution_confidence, 4),
            "ambiguity": round(self.ambiguity, 4),
        }


def derive_self_other_attribution_state(
    *,
    previous_state: Mapping[str, Any] | SelfOtherAttributionState | None = None,
    camera_observation: Mapping[str, Any] | None = None,
    subjective_scene_state: Mapping[str, Any] | None = None,
    self_state: Mapping[str, Any] | None = None,
    person_registry: Mapping[str, Any] | None = None,
) -> SelfOtherAttributionState:
    previous = coerce_self_other_attribution_state(previous_state)
    observation = dict(camera_observation or {})
    subjective_scene = dict(subjective_scene_state or {})
    self_payload = dict(self_state or {})
    registry = dict(person_registry or {})

    if not observation and not subjective_scene and not self_payload and not registry:
        return previous

    appearance_match = _clamp01(
        _float01(_pick(observation, "appearance_match", "visual_self_match", "body_appearance_match")) * 0.64
        + _float01(subjective_scene.get("self_related_salience")) * 0.18
        + _float01(subjective_scene.get("familiarity")) * 0.1
        + _float01(_pick(registry, "confidence", "self_match_confidence")) * 0.08
    )
    contingency_match = _clamp01(
        _float01(_pick(observation, "contingency_match", "temporal_contingency", "response_contingency")) * 0.72
        + _float01(subjective_scene.get("workspace_proximity")) * 0.1
        + _float01(_pick(observation, "touch_contingency", "timing_overlap")) * 0.18
    )
    perspective_match = _clamp01(
        _float01(_pick(observation, "perspective_match", "egocentric_perspective_match", "viewpoint_match")) * 0.68
        + _float01(subjective_scene.get("frontal_alignment")) * 0.18
        + _float01(subjective_scene.get("egocentric_closeness")) * 0.14
    )
    sensorimotor_consistency = _clamp01(
        _float01(_pick(observation, "sensorimotor_consistency", "motor_alignment", "agency_match")) * 0.7
        + contingency_match * 0.16
        + perspective_match * 0.14
    )

    shared_reference = _clamp01(
        _float01(_pick(observation, "shared_reference_score", "shared_attention_hint", "co_presence_hint")) * 0.58
        + _float01(subjective_scene.get("shared_scene_potential")) * 0.28
        + perspective_match * 0.14
    )
    other_reference = _clamp01(
        _float01(_pick(observation, "other_reference_score", "partner_reference_score", "otherness_hint")) * 0.64
        + max(0.0, 1.0 - sensorimotor_consistency) * 0.14
        + max(0.0, 1.0 - contingency_match) * 0.1
        + _float01(_pick(registry, "other_presence", "person_count_signal")) * 0.12
    )
    uncertainty = _clamp01(
        _float01(_pick(observation, "uncertainty", "identity_uncertainty")) * 0.46
        + _float01(subjective_scene.get("uncertainty")) * 0.22
        + _float01(self_payload.get("uncertainty")) * 0.18
        + max(0.0, 1.0 - shared_reference) * 0.14
    )

    self_score = max(
        0.0,
        appearance_match * 0.26
        + contingency_match * 0.24
        + perspective_match * 0.2
        + sensorimotor_consistency * 0.2
        + _float01(subjective_scene.get("self_related_salience")) * 0.1,
    )
    shared_score = max(
        0.0,
        shared_reference * 0.36
        + perspective_match * 0.14
        + max(contingency_match, sensorimotor_consistency) * 0.16
        + _float01(subjective_scene.get("workspace_proximity")) * 0.12
        + _float01(subjective_scene.get("comfort")) * 0.1
        + _float01(subjective_scene.get("familiarity")) * 0.12,
    )
    other_score = max(
        0.0,
        other_reference * 0.4
        + max(0.0, 1.0 - contingency_match) * 0.14
        + max(0.0, 1.0 - sensorimotor_consistency) * 0.14
        + max(0.0, 1.0 - appearance_match) * 0.08
        + _float01(_pick(registry, "other_presence", "person_count_signal")) * 0.12
        + _float01(subjective_scene.get("motion_salience")) * 0.12,
    )
    unknown_score = max(
        0.0,
        uncertainty * 0.56
        + max(0.0, 1.0 - max(self_score, shared_score, other_score)) * 0.22
        + _float01(subjective_scene.get("tension")) * 0.12
        + max(0.0, 1.0 - perspective_match) * 0.1,
    )

    self_likelihood, other_likelihood, shared_likelihood, unknown_likelihood = _normalize(
        self_score,
        other_score,
        shared_score,
        unknown_score,
    )
    ambiguity = _ambiguity(self_likelihood, other_likelihood, shared_likelihood, unknown_likelihood)
    attribution_confidence = _clamp01(max(self_likelihood, other_likelihood, shared_likelihood) * (1.0 - ambiguity * 0.45))

    appearance_match = _carry(previous.appearance_match, appearance_match, previous_state, 0.2)
    contingency_match = _carry(previous.contingency_match, contingency_match, previous_state, 0.2)
    perspective_match = _carry(previous.perspective_match, perspective_match, previous_state, 0.18)
    sensorimotor_consistency = _carry(previous.sensorimotor_consistency, sensorimotor_consistency, previous_state, 0.18)
    self_likelihood = _carry(previous.self_likelihood, self_likelihood, previous_state, 0.16)
    other_likelihood = _carry(previous.other_likelihood, other_likelihood, previous_state, 0.16)
    shared_likelihood = _carry(previous.shared_likelihood, shared_likelihood, previous_state, 0.18)
    unknown_likelihood = _carry(previous.unknown_likelihood, unknown_likelihood, previous_state, 0.18)
    ambiguity = _carry(previous.ambiguity, ambiguity, previous_state, 0.18)
    attribution_confidence = _carry(previous.attribution_confidence, attribution_confidence, previous_state, 0.18)

    dominant_attribution = _dominant_attribution(
        self_likelihood=self_likelihood,
        other_likelihood=other_likelihood,
        shared_likelihood=shared_likelihood,
        unknown_likelihood=unknown_likelihood,
        ambiguity=ambiguity,
    )
    return SelfOtherAttributionState(
        dominant_attribution=dominant_attribution,
        self_likelihood=self_likelihood,
        other_likelihood=other_likelihood,
        shared_likelihood=shared_likelihood,
        unknown_likelihood=unknown_likelihood,
        appearance_match=appearance_match,
        contingency_match=contingency_match,
        perspective_match=perspective_match,
        sensorimotor_consistency=sensorimotor_consistency,
        attribution_confidence=attribution_confidence,
        ambiguity=ambiguity,
    )


def coerce_self_other_attribution_state(
    value: Mapping[str, Any] | SelfOtherAttributionState | None,
) -> SelfOtherAttributionState:
    if isinstance(value, SelfOtherAttributionState):
        return value
    payload = dict(value or {})
    return SelfOtherAttributionState(
        dominant_attribution=_text(payload.get("dominant_attribution")) or "unknown",
        self_likelihood=_float01(payload.get("self_likelihood")),
        other_likelihood=_float01(payload.get("other_likelihood")),
        shared_likelihood=_float01(payload.get("shared_likelihood")),
        unknown_likelihood=_float01(payload.get("unknown_likelihood", 1.0)),
        appearance_match=_float01(payload.get("appearance_match")),
        contingency_match=_float01(payload.get("contingency_match")),
        perspective_match=_float01(payload.get("perspective_match")),
        sensorimotor_consistency=_float01(payload.get("sensorimotor_consistency")),
        attribution_confidence=_float01(payload.get("attribution_confidence")),
        ambiguity=_float01(payload.get("ambiguity", 1.0)),
    )


def _normalize(*scores: float) -> tuple[float, float, float, float]:
    total = sum(max(0.0, score) for score in scores)
    if total <= 1e-8:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(_clamp01(score / total) for score in scores)  # type: ignore[return-value]


def _ambiguity(*scores: float) -> float:
    ordered = sorted((_clamp01(score) for score in scores), reverse=True)
    if len(ordered) < 2:
        return 1.0
    gap = max(0.0, ordered[0] - ordered[1])
    return _clamp01(1.0 - gap)


def _dominant_attribution(
    *,
    self_likelihood: float,
    other_likelihood: float,
    shared_likelihood: float,
    unknown_likelihood: float,
    ambiguity: float,
) -> str:
    if unknown_likelihood >= max(self_likelihood, other_likelihood, shared_likelihood, 0.36) and ambiguity >= 0.56:
        return "unknown"
    if shared_likelihood >= max(self_likelihood, other_likelihood, 0.34):
        return "shared"
    if self_likelihood >= max(other_likelihood, 0.34):
        return "self"
    if other_likelihood >= 0.3:
        return "other"
    return "unknown"


def _pick(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _carry(
    previous_value: float,
    current_value: float,
    previous_state: Mapping[str, Any] | SelfOtherAttributionState | None,
    carry_ratio: float,
) -> float:
    if previous_state is None:
        return _clamp01(current_value)
    if isinstance(previous_state, Mapping) and not previous_state:
        return _clamp01(current_value)
    return _clamp01(previous_value * carry_ratio + current_value * (1.0 - carry_ratio))


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return float(numeric)


def _float01(value: Any, default: float = 0.0) -> float:
    return _clamp01(_float(value, default))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
