from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .persona_memory_fragment import PersonaMemoryFragment


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


def _tokens(values: Sequence[str]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        text = _text(value).lower()
        if not text:
            continue
        normalized = (
            text.replace(":", " ")
            .replace("/", " ")
            .replace("_", " ")
            .replace("-", " ")
        )
        for token in normalized.split():
            if token:
                tokens.add(token)
    return tokens


@dataclass(frozen=True)
class PersonaMemorySelection:
    selected_fragment_ids: tuple[str, ...]
    selected_fragments: tuple[dict[str, Any], ...]
    dominant_fragment_id: str
    total_selected_intensity: float
    scores: dict[str, float]
    dominant_inputs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_fragment_ids": list(self.selected_fragment_ids),
            "selected_fragments": [dict(item) for item in self.selected_fragments],
            "dominant_fragment_id": self.dominant_fragment_id,
            "total_selected_intensity": round(float(self.total_selected_intensity), 4),
            "scores": {
                str(key): round(float(value), 4)
                for key, value in self.scores.items()
            },
            "dominant_inputs": list(self.dominant_inputs),
        }


def derive_persona_memory_selection(
    *,
    fragments: Sequence[PersonaMemoryFragment],
    current_focus: str,
    reportable_facts: Sequence[str],
    current_risks: Sequence[str],
    relation_bias_strength: float,
    agenda_window_state: Mapping[str, Any],
    social_topology_state: Mapping[str, Any],
    grice_guard_state: Mapping[str, Any],
) -> PersonaMemorySelection:
    agenda_window = dict(agenda_window_state or {})
    topology = dict(social_topology_state or {})
    grice = dict(grice_guard_state or {})
    focus_tokens = _tokens([current_focus, *[str(item) for item in reportable_facts]])
    topology_state = _text(topology.get("state"))
    grice_state = _text(grice.get("state"))
    window_state = _text(agenda_window.get("state"))
    risk_pressure = 1.0 if list(current_risks) else 0.0
    knownness_pressure = _float01(grice.get("knownness_pressure"))
    relation_bias = _float01(relation_bias_strength)

    scores: dict[str, float] = {}
    ranked: list[tuple[float, PersonaMemoryFragment]] = []
    for fragment in fragments:
        fragment_tokens = _tokens(
            [fragment.summary, fragment.anchor, fragment.kind, *fragment.tags]
        )
        overlap = 0.0
        if focus_tokens and fragment_tokens:
            overlap = len(focus_tokens & fragment_tokens) / float(max(1, min(len(focus_tokens), len(fragment_tokens))))
        score = fragment.intensity * 0.42 + overlap * 0.32
        if fragment.kind == "relation":
            score += relation_bias * 0.24
        if fragment.kind == "theme" and window_state in {
            "next_same_group_window",
            "next_same_culture_window",
            "opportunistic_reentry",
        }:
            score += 0.16
        if fragment.kind == "boundary":
            score += max(risk_pressure * 0.22, knownness_pressure * 0.18)
            if topology_state in {"public_visible", "hierarchical"}:
                score += 0.14
        if fragment.kind == "style" and topology_state in {"one_to_one", "threaded_group"}:
            score += 0.08
        if fragment.kind == "identity" and window_state in {"revisit", "next_private_window"}:
            score += 0.08
        score = _float01(score)
        scores[fragment.fragment_id] = score
        ranked.append((score, fragment))

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected: list[dict[str, Any]] = [
        fragment.to_dict()
        for score, fragment in ranked
        if score >= 0.24
    ][:3]
    dominant_fragment_id = selected[0]["fragment_id"] if selected else ""
    total_selected_intensity = sum(
        float(item.get("intensity") or 0.0)
        for item in selected
    )
    dominant_inputs = tuple(
        item
        for item in (
            "focus_overlap" if focus_tokens else "",
            "relation_bias" if relation_bias >= 0.24 else "",
            "boundary_risk" if risk_pressure >= 1.0 else "",
            "grice_knownness" if knownness_pressure >= 0.24 else "",
            "social_topology" if topology_state else "",
            "agenda_window" if window_state else "",
        )
        if item
    )
    return PersonaMemorySelection(
        selected_fragment_ids=tuple(str(item.get("fragment_id") or "") for item in selected if str(item.get("fragment_id") or "").strip()),
        selected_fragments=tuple(selected),
        dominant_fragment_id=dominant_fragment_id,
        total_selected_intensity=_float01(total_selected_intensity),
        scores=scores,
        dominant_inputs=dominant_inputs,
    )
