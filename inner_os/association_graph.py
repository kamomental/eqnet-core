from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence

from .dot_seed import DotSeed, DotSeedSet


@dataclass(frozen=True)
class AssociationLink:
    link_id: str
    link_key: str
    left_seed_id: str
    right_seed_id: str
    left_seed_key: str
    right_seed_key: str
    weight: float
    source_diversity: float
    anchor_overlap: float
    novelty_gain: float
    unresolved_relief: float
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "link_id": self.link_id,
            "link_key": self.link_key,
            "left_seed_id": self.left_seed_id,
            "right_seed_id": self.right_seed_id,
            "left_seed_key": self.left_seed_key,
            "right_seed_key": self.right_seed_key,
            "weight": float(self.weight),
            "source_diversity": float(self.source_diversity),
            "anchor_overlap": float(self.anchor_overlap),
            "novelty_gain": float(self.novelty_gain),
            "unresolved_relief": float(self.unresolved_relief),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class AssociationGraphState:
    link_weights: Dict[str, float]
    link_counts: Dict[str, int]
    dominant_link_key: str = ""
    dominant_weight: float = 0.0
    winner_margin: float = 0.0
    dominant_inputs: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "link_weights": {str(key): float(value) for key, value in self.link_weights.items()},
            "link_counts": {str(key): int(value) for key, value in self.link_counts.items()},
            "dominant_link_key": self.dominant_link_key,
            "dominant_weight": float(self.dominant_weight),
            "winner_margin": float(self.winner_margin),
            "dominant_inputs": list(self.dominant_inputs),
        }


@dataclass(frozen=True)
class AssociationGraph:
    edges: tuple[AssociationLink, ...] = ()
    dominant_link_id: str = ""
    dominant_weight: float = 0.0
    winner_margin: float = 0.0
    dominant_inputs: tuple[str, ...] = ()
    state_hint: AssociationGraphState = field(
        default_factory=lambda: AssociationGraphState(link_weights={}, link_counts={})
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edges": [edge.to_dict() for edge in self.edges],
            "dominant_link_id": self.dominant_link_id,
            "dominant_weight": float(self.dominant_weight),
            "winner_margin": float(self.winner_margin),
            "dominant_inputs": list(self.dominant_inputs),
            "state_hint": self.state_hint.to_dict(),
        }


@dataclass
class BasicAssociationGraph:
    link_threshold: float = 0.26
    prior_gain: float = 0.18

    def build(
        self,
        *,
        dot_seeds: DotSeedSet | Sequence[DotSeed],
        previous_state: AssociationGraphState | Mapping[str, Any] | None = None,
        association_reweighting_bias: float = 0.0,
        insight_reframing_bias: float = 0.0,
        insight_class_focus: str = "",
    ) -> AssociationGraph:
        seeds = tuple(dot_seeds.seeds if isinstance(dot_seeds, DotSeedSet) else dot_seeds)
        state = coerce_association_graph_state(previous_state)
        edges: list[AssociationLink] = []
        merged_weights = dict(state.link_weights)
        merged_counts = dict(state.link_counts)
        association_reweighting_bias = _clamp01(association_reweighting_bias)
        insight_reframing_bias = _clamp01(insight_reframing_bias)
        insight_class_focus = str(insight_class_focus or "").strip()

        for left_index, left in enumerate(seeds):
            for right in seeds[left_index + 1 :]:
                edge = self._build_link(
                    left,
                    right,
                    state,
                    association_reweighting_bias=association_reweighting_bias,
                    insight_reframing_bias=insight_reframing_bias,
                    insight_class_focus=insight_class_focus,
                )
                if edge is None:
                    continue
                edges.append(edge)
                merged_weights[edge.link_key] = max(merged_weights.get(edge.link_key, 0.0), float(edge.weight))
                merged_counts.setdefault(edge.link_key, int(state.link_counts.get(edge.link_key, 0)))

        edges = sorted(edges, key=lambda item: item.weight, reverse=True)[:6]
        dominant = edges[0] if edges else None
        winner_margin = _derive_winner_margin([edge.weight for edge in edges])
        dominant_inputs = _derive_graph_dominant_inputs(
            dominant=dominant,
            association_reweighting_bias=association_reweighting_bias,
            insight_reframing_bias=insight_reframing_bias,
            insight_class_focus=insight_class_focus,
        )
        return AssociationGraph(
            edges=tuple(edges),
            dominant_link_id=dominant.link_id if dominant is not None else "",
            dominant_weight=float(dominant.weight) if dominant is not None else 0.0,
            winner_margin=winner_margin,
            dominant_inputs=dominant_inputs,
            state_hint=AssociationGraphState(
                link_weights=merged_weights,
                link_counts=merged_counts,
                dominant_link_key=dominant.link_key if dominant is not None else "",
                dominant_weight=float(dominant.weight) if dominant is not None else 0.0,
                winner_margin=winner_margin,
                dominant_inputs=dominant_inputs,
            ),
        )

    def _build_link(
        self,
        left: DotSeed,
        right: DotSeed,
        previous_state: AssociationGraphState,
        *,
        association_reweighting_bias: float,
        insight_reframing_bias: float,
        insight_class_focus: str,
    ) -> AssociationLink | None:
        link_key = _pair_key(left.seed_key, right.seed_key)
        source_diversity = 1.0 if left.source != right.source else 0.0
        anchor_overlap = _anchor_overlap(left, right)
        novelty_gain = min(float(left.novelty), float(right.novelty))
        unresolved_relief = max(float(left.unresolved_pull), float(right.unresolved_pull))
        prior = _clamp01(previous_state.link_weights.get(link_key, 0.0))
        focus_bonus = _link_focus_bonus(
            left=left,
            right=right,
            insight_class_focus=insight_class_focus,
        )
        weight = _clamp01(
            min(left.strength, right.strength) * 0.38
            + max(left.strength, right.strength) * 0.12
            + source_diversity * 0.14
            + anchor_overlap * 0.14
            + novelty_gain * 0.12
            + unresolved_relief * 0.1
            + prior * self.prior_gain * (1.0 + association_reweighting_bias * 0.5)
            + association_reweighting_bias * focus_bonus * 0.08
            + insight_reframing_bias * focus_bonus * 0.06
        )
        if weight < self.link_threshold:
            return None
        reasons: list[str] = []
        if source_diversity > 0.0:
            reasons.append("source_diversity")
        if anchor_overlap > 0.0:
            reasons.append("anchor_overlap")
        if novelty_gain >= 0.24:
            reasons.append("novelty_gain")
        if unresolved_relief >= 0.2:
            reasons.append("unresolved_relief")
        if prior >= 0.12:
            reasons.append("association_memory")
        return AssociationLink(
            link_id=f"{link_key}:{len(reasons)}",
            link_key=link_key,
            left_seed_id=left.seed_id,
            right_seed_id=right.seed_id,
            left_seed_key=left.seed_key,
            right_seed_key=right.seed_key,
            weight=weight,
            source_diversity=source_diversity,
            anchor_overlap=anchor_overlap,
            novelty_gain=novelty_gain,
            unresolved_relief=unresolved_relief,
            reasons=tuple(reasons),
        )


def apply_association_reinforcement(
    previous_state: AssociationGraphState | Mapping[str, Any] | None,
    insight_event: Mapping[str, Any] | None,
    *,
    learning_rate: float = 0.12,
    decay: float = 0.97,
    association_reweighting_focus: str = "",
    association_reweighting_reason: str = "",
    commitment_followup_focus: str = "",
    commitment_carry_bias: float = 0.0,
) -> AssociationGraphState:
    state = coerce_association_graph_state(previous_state)
    weights = {
        str(key): _clamp01(float(value) * decay)
        for key, value in state.link_weights.items()
        if _clamp01(float(value) * decay) > 1.0e-4
    }
    counts = {str(key): int(value) for key, value in state.link_counts.items()}
    event = dict(insight_event or {})
    if not bool(event.get("triggered", False)):
        dominant_link_key, dominant_weight, winner_margin = _derive_state_winner(weights)
        dominant_inputs = ("decay_only",) if dominant_link_key else ()
        return AssociationGraphState(
            link_weights=weights,
            link_counts=counts,
            dominant_link_key=dominant_link_key,
            dominant_weight=dominant_weight,
            winner_margin=winner_margin,
            dominant_inputs=dominant_inputs,
        )
    link_key = str(event.get("link_key") or "").strip()
    total_score = _clamp01(_nested_float(event, ("score", "total")))
    if link_key:
        commitment_gain = 1.0
        normalized_commitment_focus = str(commitment_followup_focus or "").strip()
        normalized_commitment_bias = _clamp01(commitment_carry_bias)
        if normalized_commitment_focus == "offer_next_step":
            commitment_gain += 0.18 * normalized_commitment_bias
        elif normalized_commitment_focus == "reopen_softly":
            commitment_gain += 0.12 * normalized_commitment_bias
        elif normalized_commitment_focus == "hold":
            commitment_gain -= 0.14 * normalized_commitment_bias
        weights[link_key] = _clamp01(
            weights.get(link_key, 0.0) + learning_rate * max(total_score, 0.12) * max(0.72, commitment_gain)
        )
        counts[link_key] = counts.get(link_key, 0) + 1
    dominant_inputs = [
        str(item)
        for item in event.get("reasons") or []
        if str(item).strip()
    ]
    if association_reweighting_focus:
        dominant_inputs.append(f"overnight_focus:{str(association_reweighting_focus).strip()}")
    if association_reweighting_reason:
        dominant_inputs.append(f"overnight_reason:{str(association_reweighting_reason).strip()}")
    if commitment_followup_focus:
        dominant_inputs.append(f"overnight_commitment:{str(commitment_followup_focus).strip()}")
    dominant_link_key, dominant_weight, winner_margin = _derive_state_winner(weights)
    return AssociationGraphState(
        link_weights=weights,
        link_counts=counts,
        dominant_link_key=dominant_link_key,
        dominant_weight=dominant_weight,
        winner_margin=winner_margin,
        dominant_inputs=tuple(dict.fromkeys(dominant_inputs)),
    )


def coerce_association_graph_state(
    value: AssociationGraphState | Mapping[str, Any] | None,
) -> AssociationGraphState:
    if isinstance(value, AssociationGraphState):
        return value
    if not isinstance(value, Mapping):
        return AssociationGraphState(link_weights={}, link_counts={})
    raw_weights = value.get("link_weights")
    raw_counts = value.get("link_counts")
    weights = (
        {str(key): _clamp01(item) for key, item in raw_weights.items() if str(key).strip()}
        if isinstance(raw_weights, Mapping)
        else {}
    )
    counts = (
        {str(key): max(0, int(item)) for key, item in raw_counts.items() if str(key).strip()}
        if isinstance(raw_counts, Mapping)
        else {}
    )
    dominant_inputs = tuple(
        str(item)
        for item in value.get("dominant_inputs") or []
        if str(item).strip()
    )
    return AssociationGraphState(
        link_weights=weights,
        link_counts=counts,
        dominant_link_key=str(value.get("dominant_link_key") or "").strip(),
        dominant_weight=_clamp01(value.get("dominant_weight")),
        winner_margin=_clamp01(value.get("winner_margin")),
        dominant_inputs=dominant_inputs,
    )


def _pair_key(left_seed_key: str, right_seed_key: str) -> str:
    parts = sorted([str(left_seed_key).strip(), str(right_seed_key).strip()])
    return "|".join(part for part in parts if part) or "link"


def _anchor_overlap(left: DotSeed, right: DotSeed) -> float:
    if left.anchor and right.anchor and left.anchor == right.anchor:
        return 1.0
    left_tokens = _tokens(left.label) | _tokens(left.anchor)
    right_tokens = _tokens(right.label) | _tokens(right.anchor)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    if union <= 0:
        return 0.0
    return float(intersection / union)


def _link_focus_bonus(
    *,
    left: DotSeed,
    right: DotSeed,
    insight_class_focus: str,
) -> float:
    focus = str(insight_class_focus or "").strip()
    if not focus:
        return 0.0
    sources = {left.source, right.source}
    if focus == "reframed_relation":
        return 1.0 if "bond" in sources and ("memory" in sources or "external_cue" in sources) else 0.0
    if focus == "new_link_hypothesis":
        return 1.0 if "external_cue" in sources and len(sources) >= 2 else 0.2
    if focus == "insight_trace":
        return 0.6 if len(sources) >= 2 else 0.2
    return 0.0


def _derive_winner_margin(values: Sequence[float]) -> float:
    ordered = sorted((_clamp01(value) for value in values), reverse=True)
    if not ordered:
        return 0.0
    top = ordered[0]
    runner_up = ordered[1] if len(ordered) > 1 else 0.0
    return _clamp01(top - runner_up)


def _derive_graph_dominant_inputs(
    *,
    dominant: AssociationLink | None,
    association_reweighting_bias: float,
    insight_reframing_bias: float,
    insight_class_focus: str,
) -> tuple[str, ...]:
    if dominant is None:
        return ()
    dominant_inputs = [
        str(item)
        for item in dominant.reasons
        if str(item).strip()
    ]
    if association_reweighting_bias > 0.0 and "association_memory" in dominant_inputs:
        dominant_inputs.append("overnight_association_bias")
    if insight_reframing_bias > 0.0 and str(insight_class_focus or "").strip():
        dominant_inputs.append(f"insight_focus:{str(insight_class_focus).strip()}")
    if not dominant_inputs:
        dominant_inputs.append("link_weight")
    return tuple(dict.fromkeys(dominant_inputs))


def _derive_state_winner(weights: Mapping[str, float]) -> tuple[str, float, float]:
    ordered = sorted(
        ((str(key), _clamp01(value)) for key, value in weights.items() if str(key).strip()),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ordered:
        return "", 0.0, 0.0
    dominant_link_key, dominant_weight = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
    return dominant_link_key, dominant_weight, _clamp01(dominant_weight - runner_up)


def _tokens(value: str) -> set[str]:
    token = []
    tokens: set[str] = set()
    for char in str(value or "").lower():
        if char.isalnum():
            token.append(char)
        elif token:
            tokens.add("".join(token))
            token = []
    if token:
        tokens.add("".join(token))
    return tokens


def _nested_float(payload: Mapping[str, Any], path: tuple[str, ...]) -> float:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping):
            return 0.0
        current = current.get(key)
    return _clamp01(current)


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))
