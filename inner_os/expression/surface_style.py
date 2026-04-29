from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SurfaceStyleDecision:
    style_id: str
    reason: str
    warmth: float
    playfulness: float
    restraint: float

    def to_dict(self) -> dict[str, object]:
        return {
            "style_id": self.style_id,
            "reason": self.reason,
            "warmth": round(self.warmth, 4),
            "playfulness": round(self.playfulness, 4),
            "restraint": round(self.restraint, 4),
        }


@dataclass(frozen=True)
class SurfaceStyleConfig:
    friendly_threshold: float = 0.42
    warm_playful_threshold: float = 0.58
    restraint_plain_threshold: float = 0.62


def derive_surface_style_decision(
    *,
    audit_projection: Mapping[str, Any] | None = None,
    joint_state: Mapping[str, Any] | None = None,
    shared_presence: Mapping[str, Any] | None = None,
    default_style: str = "plain",
    config: SurfaceStyleConfig | None = None,
) -> SurfaceStyleDecision:
    active_config = config or SurfaceStyleConfig()
    audit = dict(audit_projection or {})
    axes = dict(audit.get("audit_axes") or {})
    expression_context = dict(audit.get("expression_context_state") or {})
    culture = dict(expression_context.get("culture") or {})
    relation = dict(expression_context.get("relational_style") or {})
    shared_topic = dict(expression_context.get("shared_topic") or {})
    joint = dict(joint_state or audit.get("joint_state") or {})
    shared = dict(shared_presence or audit.get("shared_presence") or {})
    source_state = dict(audit.get("surface_context_source_state") or {})

    restraint = _clamp01(
        _max_value(
            axes.get("culture_politeness_pressure"),
            culture.get("politeness_pressure"),
            axes.get("context_surface_caution"),
            axes.get("organism_protective_tension"),
            axes.get("protective_trace_stabilization_need"),
        )
    )
    warmth = _clamp01(
        _weighted_sum(
            (
                (_float01(joint.get("common_ground") or axes.get("joint_common_ground")), 0.22),
                (_float01(joint.get("shared_delight")), 0.22),
                (_float01(shared.get("co_presence")), 0.16),
                (_float01(shared_topic.get("affinity") or culture.get("shared_topic_affinity")), 0.18),
                (_float01(culture.get("shared_interest_affinity") or culture.get("hobby_resonance")), 0.18),
                (_float01(culture.get("culture_resonance") or axes.get("culture_culture_resonance")), 0.04),
            )
        )
    )
    playfulness = _clamp01(
        _weighted_sum(
            (
                (_float01(culture.get("joke_ratio_ceiling") or axes.get("culture_joke_ratio_ceiling")), 0.34),
                (_float01(relation.get("playful_ceiling") or relation.get("banter_room")), 0.22),
                (_float01(shared_topic.get("playfulness")), 0.12),
                (_float01(source_state.get("utterance_reason_offer") == "brief_shared_smile"), 0.14),
                (_float01(joint.get("shared_delight")), 0.18),
            )
        )
    )

    if restraint >= active_config.restraint_plain_threshold:
        return SurfaceStyleDecision(
            style_id="plain",
            reason="restraint_overrides_style",
            warmth=warmth,
            playfulness=playfulness,
            restraint=restraint,
        )
    if (
        warmth >= active_config.warm_playful_threshold
        and playfulness >= active_config.warm_playful_threshold
    ):
        return SurfaceStyleDecision(
            style_id="warm_playful",
            reason="shared_warmth_and_playfulness",
            warmth=warmth,
            playfulness=playfulness,
            restraint=restraint,
        )
    if warmth >= active_config.friendly_threshold or playfulness >= active_config.friendly_threshold:
        return SurfaceStyleDecision(
            style_id="friendly",
            reason="shared_affinity_or_lightness",
            warmth=warmth,
            playfulness=playfulness,
            restraint=restraint,
        )
    style = str(default_style or "plain").strip() or "plain"
    return SurfaceStyleDecision(
        style_id=style,
        reason="default_style",
        warmth=warmth,
        playfulness=playfulness,
        restraint=restraint,
    )


def _weighted_sum(items: tuple[tuple[float, float], ...]) -> float:
    return sum(value * weight for value, weight in items)


def _max_value(*values: Any) -> float:
    return max((_float01(value) for value in values), default=0.0)


def _float01(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return _clamp01(numeric)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
