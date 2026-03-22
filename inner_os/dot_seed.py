from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np


@dataclass(frozen=True)
class DotSeed:
    seed_id: str
    seed_key: str
    label: str
    source: str
    anchor: str
    strength: float
    novelty: float
    unresolved_pull: float
    cues: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed_id": self.seed_id,
            "seed_key": self.seed_key,
            "label": self.label,
            "source": self.source,
            "anchor": self.anchor,
            "strength": float(self.strength),
            "novelty": float(self.novelty),
            "unresolved_pull": float(self.unresolved_pull),
            "cues": list(self.cues),
        }


@dataclass(frozen=True)
class DotSeedSet:
    seeds: tuple[DotSeed, ...] = ()
    dominant_seed_id: str = ""
    dominant_seed_label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seeds": [seed.to_dict() for seed in self.seeds],
            "dominant_seed_id": self.dominant_seed_id,
            "dominant_seed_label": self.dominant_seed_label,
        }


def derive_dot_seeds(
    *,
    qualia_state: Mapping[str, Any] | None,
    current_state: Mapping[str, Any] | None,
    current_text: str = "",
    current_focus: str = "",
    max_felt_seeds: int = 2,
) -> DotSeedSet:
    qualia = dict(qualia_state or {})
    state = dict(current_state or {})
    text = str(current_text or "").strip()
    focus = str(current_focus or state.get("current_focus") or "").strip()
    seeds: list[DotSeed] = []

    seeds.extend(_felt_seeds(qualia, focus=focus, max_items=max_felt_seeds))

    memory_seed = _memory_seed(state, focus=focus)
    if memory_seed is not None:
        seeds.append(memory_seed)

    unresolved_seed = _unresolved_seed(state, focus=focus)
    if unresolved_seed is not None:
        seeds.append(unresolved_seed)

    bond_seed = _bond_seed(state, focus=focus)
    if bond_seed is not None:
        seeds.append(bond_seed)

    external_seed = _external_seed(text=text, focus=focus)
    if external_seed is not None:
        seeds.append(external_seed)

    seeds = sorted(seeds, key=lambda item: item.strength, reverse=True)[:6]
    dominant = seeds[0] if seeds else None
    return DotSeedSet(
        seeds=tuple(seeds),
        dominant_seed_id=dominant.seed_id if dominant is not None else "",
        dominant_seed_label=dominant.label if dominant is not None else "",
    )


def _felt_seeds(
    qualia_state: Mapping[str, Any],
    *,
    focus: str,
    max_items: int,
) -> list[DotSeed]:
    qualia_values = _float_list(qualia_state.get("qualia"))
    gate_values = _float_list(qualia_state.get("gate"))
    habituation = _float_list(qualia_state.get("habituation"))
    value_grad = _float_list(qualia_state.get("value_grad"))
    axis_labels = [str(item).strip() for item in qualia_state.get("axis_labels") or [] if str(item).strip()]
    trust = _clamp01(qualia_state.get("trust_applied"))
    if not qualia_values or not gate_values:
        return []
    size = min(len(qualia_values), len(gate_values))
    habituation_norm = _normalize(habituation, size)
    value_norm = _normalize(value_grad, size)
    ranked: list[tuple[int, float]] = []
    for index in range(size):
        strength = abs(qualia_values[index]) * max(gate_values[index], 0.0) * max(trust, 0.12)
        ranked.append((index, strength))
    ranked.sort(key=lambda item: item[1], reverse=True)

    seeds: list[DotSeed] = []
    for rank, (index, strength) in enumerate(ranked[: max(1, max_items)]):
        if strength <= 1.0e-4:
            continue
        label = axis_labels[index] if index < len(axis_labels) else f"felt_axis_{index}"
        key = f"felt:{_normalize_token(label)}"
        novelty = _clamp01(0.14 + (1.0 - habituation_norm[index]) * 0.56)
        unresolved_pull = _clamp01(value_norm[index] * 0.42)
        seeds.append(
            DotSeed(
                seed_id=f"{key}:{rank}",
                seed_key=key,
                label=label,
                source="felt",
                anchor=focus or label,
                strength=_clamp01(strength),
                novelty=novelty,
                unresolved_pull=unresolved_pull,
                cues=("dot_seed:felt",),
            )
        )
    return seeds


def _memory_seed(state: Mapping[str, Any], *, focus: str) -> DotSeed | None:
    label = next(
        (
            str(value).strip()
            for value in (
                state.get("long_term_theme_summary"),
                state.get("relation_seed_summary"),
                state.get("conscious_residue_summary"),
                state.get("memory_anchor"),
            )
            if str(value or "").strip()
        ),
        "",
    )
    if not label:
        return None
    strength = _clamp01(
        max(
            _float(state.get("replay_intensity")),
            _float(state.get("meaning_inertia")),
            _float(state.get("conscious_residue_strength")),
        )
    )
    if strength <= 0.08:
        return None
    anchor = str(state.get("memory_anchor") or focus or label).strip()
    key = f"memory:{_normalize_token(anchor or label)}"
    novelty = _clamp01(0.1 + _float(state.get("interaction_afterglow")) * 0.22)
    unresolved_pull = _clamp01(_float(state.get("pending_meaning")) * 0.4)
    return DotSeed(
        seed_id=f"{key}:0",
        seed_key=key,
        label=label[:80],
        source="memory",
        anchor=anchor,
        strength=strength,
        novelty=novelty,
        unresolved_pull=unresolved_pull,
        cues=("dot_seed:memory",),
    )


def _unresolved_seed(state: Mapping[str, Any], *, focus: str) -> DotSeed | None:
    unresolved_count = max(0, int(state.get("unresolved_count") or 0))
    pending_meaning = _clamp01(state.get("pending_meaning"))
    if unresolved_count <= 0 and pending_meaning <= 0.08:
        return None
    label = focus or str(state.get("conscious_residue_focus") or "unresolved_thread").strip()
    key = f"unresolved:{_normalize_token(label)}"
    strength = _clamp01(pending_meaning * 0.72 + min(unresolved_count, 3) * 0.12)
    novelty = _clamp01(0.18 + pending_meaning * 0.18)
    return DotSeed(
        seed_id=f"{key}:0",
        seed_key=key,
        label=label[:80],
        source="unresolved",
        anchor=label[:80],
        strength=strength,
        novelty=novelty,
        unresolved_pull=_clamp01(0.28 + pending_meaning * 0.52),
        cues=("dot_seed:unresolved",),
    )


def _bond_seed(state: Mapping[str, Any], *, focus: str) -> DotSeed | None:
    person_id = str(state.get("related_person_id") or "").strip()
    attachment = _clamp01(state.get("attachment"))
    trust_memory = _clamp01(state.get("trust_memory"))
    if not person_id and max(attachment, trust_memory) <= 0.18:
        return None
    label = person_id or focus or "bond"
    key = f"bond:{_normalize_token(label)}"
    strength = _clamp01(attachment * 0.58 + trust_memory * 0.42)
    novelty = _clamp01(0.12 + _clamp01(state.get("social_grounding")) * 0.12)
    unresolved_pull = _clamp01(_float(state.get("recent_strain")) * 0.18)
    return DotSeed(
        seed_id=f"{key}:0",
        seed_key=key,
        label=label[:80],
        source="bond",
        anchor=label[:80],
        strength=max(strength, 0.12),
        novelty=novelty,
        unresolved_pull=unresolved_pull,
        cues=("dot_seed:bond",),
    )


def _external_seed(*, text: str, focus: str) -> DotSeed | None:
    label = focus or text
    if not label:
        return None
    summary = text[:80] if text else label[:80]
    text_scale = min(len(text.strip()), 80) / 80.0 if text else 0.0
    strength = _clamp01(0.26 + text_scale * 0.38 + (0.08 if focus else 0.0))
    key = f"external:{_normalize_token(focus or summary)}"
    return DotSeed(
        seed_id=f"{key}:0",
        seed_key=key,
        label=(focus or summary)[:80],
        source="external_cue",
        anchor=(focus or summary)[:80],
        strength=strength,
        novelty=_clamp01(0.34 + text_scale * 0.32),
        unresolved_pull=0.0,
        cues=("dot_seed:external",),
    )


def _float_list(values: Any) -> list[float]:
    if values is None:
        return []
    result: list[float] = []
    for item in values:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            result.append(0.0)
    return result


def _normalize(values: list[float], size: int) -> list[float]:
    if size <= 0:
        return []
    clipped = [max(0.0, float(item)) for item in values[:size]]
    if len(clipped) < size:
        clipped.extend([0.0] * (size - len(clipped)))
    scale = max(clipped) if clipped else 0.0
    if scale <= 1.0e-6:
        return [0.0] * size
    return [item / scale for item in clipped]


def _normalize_token(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "").strip())
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "seed"


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, _float(value)))
