from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .group_thread_registry import summarize_group_thread_registry_snapshot


@dataclass(frozen=True)
class SocialTopologyState:
    state: str
    score: float
    winner_margin: float
    scores: dict[str, float]
    visibility_pressure: float
    threading_pressure: float
    hierarchy_pressure: float
    total_people: int
    raw_topology_hint: str
    dominant_inputs: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "score": round(self.score, 4),
            "winner_margin": round(self.winner_margin, 4),
            "scores": {key: round(value, 4) for key, value in self.scores.items()},
            "visibility_pressure": round(self.visibility_pressure, 4),
            "threading_pressure": round(self.threading_pressure, 4),
            "hierarchy_pressure": round(self.hierarchy_pressure, 4),
            "total_people": int(self.total_people),
            "raw_topology_hint": self.raw_topology_hint,
            "dominant_inputs": list(self.dominant_inputs),
        }


def derive_social_topology_state(
    *,
    scene_state: Mapping[str, Any] | None,
    relation_competition_state: Mapping[str, Any] | None,
    related_person_ids: Sequence[str] | None,
    self_state: Mapping[str, Any] | None = None,
) -> SocialTopologyState:
    scene = dict(scene_state or {})
    relation_competition = dict(relation_competition_state or {})
    self_payload = dict(self_state or {})
    registry_snapshot = dict(self_payload.get("person_registry_snapshot") or {})
    group_thread_registry_snapshot = dict(self_payload.get("group_thread_registry_snapshot") or {})
    group_thread_registry_summary = summarize_group_thread_registry_snapshot(
        group_thread_registry_snapshot
    )
    group_thread_focus = str(self_payload.get("group_thread_focus") or "").strip()
    group_thread_carry_bias = _clamp01(self_payload.get("group_thread_carry_bias"))
    raw_topology_hint = str(
        scene.get("social_topology")
        or self_payload.get("social_topology")
        or ""
    ).strip()
    scene_tags = {
        str(item).strip()
        for item in scene.get("scene_tags") or []
        if str(item).strip()
    }
    privacy_level = _clamp01(scene.get("privacy_level"))
    norm_pressure = _clamp01(scene.get("norm_pressure"))
    environmental_load = _clamp01(scene.get("environmental_load"))
    competition_state = str(relation_competition.get("state") or "ambient").strip() or "ambient"
    competition_level = _clamp01(relation_competition.get("competition_level"))
    competition_margin = _clamp01(relation_competition.get("winner_margin"))
    related_people = [
        str(item).strip()
        for item in related_person_ids or []
        if str(item).strip()
    ]
    registry_total_people = int(registry_snapshot.get("total_people") or 0)
    total_people = max(
        len(related_people),
        int(relation_competition.get("total_people") or 0),
        registry_total_people,
    )
    relation_active = 1.0 if total_people > 0 or str(relation_competition.get("dominant_person_id") or "").strip() else 0.0

    visibility_pressure = _clamp01(
        (1.0 - privacy_level) * 0.42
        + norm_pressure * 0.24
        + environmental_load * 0.12
        + (0.18 if raw_topology_hint == "public_visible" else 0.0)
        + (0.12 if "socially_exposed" in scene_tags else 0.0)
    )
    threading_pressure = _clamp01(
        competition_level * 0.54
        + (0.18 if total_people > 1 else 0.0)
        + (0.14 if raw_topology_hint in {"group_present", "multi_person"} else 0.0)
        + (0.08 if competition_state == "competing_threads" else 0.0)
        + (0.06 if competition_margin <= 0.14 and total_people > 1 else 0.0)
        + (0.06 if int(group_thread_registry_summary.get("total_threads") or 0) > 1 else 0.0)
        + (
            group_thread_carry_bias * 0.24
            if group_thread_focus == "threaded_group"
            else 0.0
        )
    )
    hierarchy_pressure = _clamp01(
        (0.64 if raw_topology_hint == "hierarchical" else 0.0)
        + norm_pressure * 0.16
        + (0.08 if "high_norm" in scene_tags else 0.0)
        + (0.06 if total_people > 1 else 0.0)
        + (
            group_thread_carry_bias * 0.22
            if group_thread_focus == "hierarchical"
            else 0.0
        )
    )
    visibility_pressure = _clamp01(
        visibility_pressure
        + (
            group_thread_carry_bias * 0.22
            if group_thread_focus == "public_visible"
            else 0.0
        )
    )

    scores = {
        "ambient": _clamp01(
            (1.0 - relation_active) * 0.62
            + (1.0 - visibility_pressure) * 0.14
            + (0.1 if total_people == 0 else 0.0)
        ),
        "one_to_one": _clamp01(
            (0.46 if total_people == 1 else 0.0)
            + (0.12 if raw_topology_hint == "one_to_one" else 0.0)
            + (1.0 - visibility_pressure) * 0.16
            + (1.0 - hierarchy_pressure) * 0.1
            + (0.1 if competition_state == "single_anchor" else 0.0)
            + (
                group_thread_carry_bias * 0.16
                if group_thread_focus == "one_to_one"
                else 0.0
            )
            - threading_pressure * 0.16
        ),
        "threaded_group": _clamp01(
            threading_pressure * 0.64
            + (0.14 if total_people > 1 else 0.0)
            + (0.08 if competition_state in {"competing_threads", "dominant_thread"} else 0.0)
            + (
                group_thread_carry_bias * 0.18
                if group_thread_focus == "threaded_group"
                else 0.0
            )
            - visibility_pressure * 0.08
        ),
        "public_visible": _clamp01(
            visibility_pressure * 0.68
            + (0.12 if raw_topology_hint == "public_visible" else 0.0)
            + (0.08 if relation_active >= 1.0 else 0.0)
            + (
                group_thread_carry_bias * 0.16
                if group_thread_focus == "public_visible"
                else 0.0
            )
            - hierarchy_pressure * 0.08
        ),
        "hierarchical": _clamp01(
            hierarchy_pressure * 0.72
            + (0.12 if raw_topology_hint == "hierarchical" else 0.0)
            + (0.06 if relation_active >= 1.0 else 0.0)
            + (
                group_thread_carry_bias * 0.16
                if group_thread_focus == "hierarchical"
                else 0.0
            )
        ),
    }
    state, winner_margin = _winner_and_margin(scores)
    dominant_inputs = _compact(
        [
            "public_visibility" if visibility_pressure >= 0.28 else "",
            "threaded_group" if threading_pressure >= 0.28 else "",
            "hierarchy_signal" if hierarchy_pressure >= 0.28 else "",
            "multiple_people" if total_people > 1 else "",
            "scene_public_visible" if raw_topology_hint == "public_visible" else "",
            "scene_hierarchical" if raw_topology_hint == "hierarchical" else "",
            "scene_group_present" if raw_topology_hint in {"group_present", "multi_person"} else "",
            "tight_relation_margin" if total_people > 1 and competition_margin <= 0.14 else "",
            "overnight_group_thread_focus" if group_thread_focus else "",
        ]
    )
    return SocialTopologyState(
        state=state,
        score=scores[state],
        winner_margin=winner_margin,
        scores=scores,
        visibility_pressure=visibility_pressure,
        threading_pressure=threading_pressure,
        hierarchy_pressure=hierarchy_pressure,
        total_people=total_people,
        raw_topology_hint=raw_topology_hint,
        dominant_inputs=dominant_inputs,
    )


def coerce_social_topology_label(topology_state_name: str) -> str:
    state = str(topology_state_name or "").strip()
    if state == "threaded_group":
        return "multi_person"
    if state in {"ambient", "one_to_one", "public_visible", "hierarchical"}:
        return state
    return "ambient"


def _winner_and_margin(scores: Mapping[str, float]) -> tuple[str, float]:
    ranked = sorted(
        ((str(key), _clamp01(value)) for key, value in scores.items()),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )
    if not ranked:
        return "ambient", 0.0
    winner = ranked[0][0]
    winner_score = ranked[0][1]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    return winner, _clamp01(winner_score - runner_up)


def _compact(values: Sequence[str]) -> list[str]:
    return [value for value in values if value]


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value or 0.0)
    except (TypeError, ValueError):
        numeric = 0.0
    if numeric <= 0.0:
        return 0.0
    if numeric >= 1.0:
        return 1.0
    return float(numeric)
