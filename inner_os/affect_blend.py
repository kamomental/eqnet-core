from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

from .interaction.models import LiveInteractionRegulation, RelationalMood, SituationState
from .scene_state import SceneState


@dataclass(frozen=True)
class AffectBlendState:
    care: float = 0.0
    reverence: float = 0.0
    innocence: float = 0.0
    defense: float = 0.0
    future_pull: float = 0.0
    shared_world_pull: float = 0.0
    distress: float = 0.0
    confidence: float = 0.0
    conflict_level: float = 0.0
    residual_tension: float = 0.0
    reportability_pressure: float = 0.0
    dominant_mode: str = "ambient"
    cues: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def derive_affect_blend_state(
    *,
    affective_summary: Mapping[str, Any],
    relational_mood: RelationalMood,
    live_regulation: LiveInteractionRegulation,
    situation_state: SituationState,
    scene_state: SceneState,
    stress: float = 0.0,
    recovery_need: float = 0.0,
    safety_bias: float = 0.0,
    relation_bias_strength: float = 0.0,
    temporal_membrane_bias: Mapping[str, Any] | None = None,
) -> AffectBlendState:
    arousal = _clamp01(float(affective_summary.get("arousal", 0.0) or 0.0))
    social_tension = _clamp01(float(affective_summary.get("social_tension", 0.0) or 0.0))
    temporal_bias = dict(temporal_membrane_bias or {})
    timeline_coherence = _clamp01(float(temporal_bias.get("timeline_coherence", 0.0) or 0.0))
    reentry_pull = _clamp01(float(temporal_bias.get("reentry_pull", 0.0) or 0.0))
    supersession_pressure = _clamp01(float(temporal_bias.get("supersession_pressure", 0.0) or 0.0))
    continuity_pressure = _clamp01(float(temporal_bias.get("continuity_pressure", 0.0) or 0.0))
    relation_reentry_pull = _clamp01(float(temporal_bias.get("relation_reentry_pull", 0.0) or 0.0))

    care = _clamp01(
        relational_mood.care * 0.62
        + relation_bias_strength * 0.14
        + situation_state.shared_attention * 0.08
        + continuity_pressure * 0.08
        + relation_reentry_pull * 0.08
    )
    reverence = _clamp01(
        relational_mood.reverence * 0.68
        + scene_state.norm_pressure * 0.22
        + (0.08 if scene_state.scene_family == "reverent_distance" else 0.0)
    )
    innocence = _clamp01(
        relational_mood.innocence * 0.72
        + max(0.0, 1.0 - relational_mood.confidence_signal) * 0.12
    )
    defense = _clamp01(
        stress * 0.28
        + safety_bias * 0.24
        + social_tension * 0.18
        + arousal * 0.12
        + live_regulation.strained_pause * 0.18
        + supersession_pressure * 0.12
    )
    future_pull = _clamp01(
        relational_mood.future_pull * 0.72
        + live_regulation.future_loop_pull * 0.22
        + reentry_pull * 0.12
        + continuity_pressure * 0.06
    )
    shared_world_pull = _clamp01(
        relational_mood.shared_world_pull * 0.74
        + situation_state.shared_attention * 0.12
        + timeline_coherence * 0.08
        + relation_reentry_pull * 0.06
    )
    distress = _clamp01(
        stress * 0.34
        + recovery_need * 0.26
        + social_tension * 0.18
        + live_regulation.strained_pause * 0.12
        + supersession_pressure * 0.08
    )
    confidence = _clamp01(
        relational_mood.confidence_signal * 0.62
        + max(0.0, 1.0 - distress) * 0.18
        + max(0.0, scene_state.safety_margin) * 0.12
        + timeline_coherence * 0.08
        - supersession_pressure * 0.06
    )

    conflict_level = _clamp01(
        min(care, defense) * 0.28
        + min(future_pull, defense) * 0.36
        + min(shared_world_pull, reverence) * 0.14
        + min(future_pull, reverence) * 0.14
        + min(shared_world_pull, distress) * 0.08
        + min(continuity_pressure, supersession_pressure) * 0.1
    )
    residual_tension = _clamp01(
        conflict_level * 0.58
        + distress * 0.18
        + live_regulation.strained_pause * 0.16
        + (0.08 if live_regulation.repair_window_open else 0.0)
        + supersession_pressure * 0.08
    )
    reportability_pressure = _clamp01(
        care * 0.16
        + future_pull * 0.14
        + shared_world_pull * 0.14
        + distress * 0.14
        + conflict_level * 0.16
        + residual_tension * 0.14
        + max(0.0, scene_state.privacy_level - scene_state.norm_pressure) * 0.12
        + reentry_pull * 0.06
    )

    mode_scores = {
        "care": care,
        "reverence": reverence,
        "innocence": innocence,
        "defense": defense,
        "future_pull": future_pull,
        "shared_world_pull": shared_world_pull,
        "distress": distress,
    }
    dominant_mode = max(mode_scores.items(), key=lambda item: item[1])[0]

    cues: list[str] = [f"blend_{dominant_mode}"]
    if conflict_level >= 0.42:
        cues.append("blend_conflict_high")
    if residual_tension >= 0.4:
        cues.append("blend_unsettled")
    if defense >= 0.42:
        cues.append("blend_defensive")
    if shared_world_pull >= 0.44:
        cues.append("blend_shared_world")
    if continuity_pressure >= 0.34:
        cues.append("blend_temporal_continuity")
    if reentry_pull >= 0.36 or relation_reentry_pull >= 0.36:
        cues.append("blend_temporal_reentry")
    if supersession_pressure >= 0.34:
        cues.append("blend_temporal_supersession")

    return AffectBlendState(
        care=round(care, 4),
        reverence=round(reverence, 4),
        innocence=round(innocence, 4),
        defense=round(defense, 4),
        future_pull=round(future_pull, 4),
        shared_world_pull=round(shared_world_pull, 4),
        distress=round(distress, 4),
        confidence=round(confidence, 4),
        conflict_level=round(conflict_level, 4),
        residual_tension=round(residual_tension, 4),
        reportability_pressure=round(reportability_pressure, 4),
        dominant_mode=dominant_mode,
        cues=tuple(cues),
    )


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))
