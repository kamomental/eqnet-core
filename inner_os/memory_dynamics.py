from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class MemoryRelationEdge:
    """記憶どうしの現在有効な関係仮説。"""

    relation_id: str
    relation_key: str
    relation_type: str
    source_key: str
    target_key: str
    weight: float
    confidence: float
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "relation_key": self.relation_key,
            "relation_type": self.relation_type,
            "source_key": self.source_key,
            "target_key": self.target_key,
            "weight": round(self.weight, 4),
            "confidence": round(self.confidence, 4),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class MemoryMetaRelation:
    """関係どうしの間に立つ meta-relation。"""

    left_relation_id: str
    right_relation_id: str
    meta_type: str
    strength: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_relation_id": self.left_relation_id,
            "right_relation_id": self.right_relation_id,
            "meta_type": self.meta_type,
            "strength": round(self.strength, 4),
        }


@dataclass(frozen=True)
class MemoryCausalEdge:
    """記憶関係から立ち上がる因果的な向きづけ。"""

    causal_id: str
    causal_key: str
    causal_type: str
    cause_key: str
    effect_key: str
    weight: float
    confidence: float
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "causal_id": self.causal_id,
            "causal_key": self.causal_key,
            "causal_type": self.causal_type,
            "cause_key": self.cause_key,
            "effect_key": self.effect_key,
            "weight": round(self.weight, 4),
            "confidence": round(self.confidence, 4),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class MemoryDynamicsFrame:
    """palace / monument / ignition / reconsolidation の時間断面。"""

    step: int = 0
    palace_mode: str = "ambient"
    monument_mode: str = "ambient"
    ignition_mode: str = "idle"
    reconsolidation_mode: str = "settle"
    recall_anchor: str = ""
    palace_topology: float = 0.0
    palace_density: float = 0.0
    monument_salience: float = 0.0
    ignition_readiness: float = 0.0
    activation_confidence: float = 0.0
    consolidation_pull: float = 0.0
    memory_tension: float = 0.0
    prospective_pull: float = 0.0
    dominant_mode: str = "stabilize"

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "palace_mode": self.palace_mode,
            "monument_mode": self.monument_mode,
            "ignition_mode": self.ignition_mode,
            "reconsolidation_mode": self.reconsolidation_mode,
            "recall_anchor": self.recall_anchor,
            "palace_topology": round(self.palace_topology, 4),
            "palace_density": round(self.palace_density, 4),
            "monument_salience": round(self.monument_salience, 4),
            "ignition_readiness": round(self.ignition_readiness, 4),
            "activation_confidence": round(self.activation_confidence, 4),
            "consolidation_pull": round(self.consolidation_pull, 4),
            "memory_tension": round(self.memory_tension, 4),
            "prospective_pull": round(self.prospective_pull, 4),
            "dominant_mode": self.dominant_mode,
        }


@dataclass(frozen=True)
class MemoryDynamicsState:
    """記憶の topology / salience / ignition / reconsolidation を束ねる canonical state。"""

    palace_topology: float = 0.0
    palace_density: float = 0.0
    palace_mode: str = "ambient"
    dominant_link_key: str = ""
    dominant_link_inputs: tuple[str, ...] = ()
    monument_salience: float = 0.0
    monument_kind: str = ""
    monument_mode: str = "ambient"
    ignition_readiness: float = 0.0
    ignition_mode: str = "idle"
    activation_confidence: float = 0.0
    recall_anchor: str = ""
    consolidation_pull: float = 0.0
    replay_priority: float = 0.0
    reconsolidation_priority: float = 0.0
    autobiographical_pull: float = 0.0
    reconsolidation_mode: str = "settle"
    forgetting_pressure: float = 0.0
    memory_tension: float = 0.0
    prospective_pull: float = 0.0
    dominant_relation_type: str = ""
    relation_generation_mode: str = "ambient"
    relation_edges: tuple[MemoryRelationEdge, ...] = ()
    meta_relations: tuple[MemoryMetaRelation, ...] = ()
    dominant_causal_type: str = ""
    causal_generation_mode: str = "ambient"
    causal_edges: tuple[MemoryCausalEdge, ...] = ()
    dominant_mode: str = "stabilize"
    trace: tuple[MemoryDynamicsFrame, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "palace_topology": round(self.palace_topology, 4),
            "palace_density": round(self.palace_density, 4),
            "palace_mode": self.palace_mode,
            "dominant_link_key": self.dominant_link_key,
            "dominant_link_inputs": list(self.dominant_link_inputs),
            "monument_salience": round(self.monument_salience, 4),
            "monument_kind": self.monument_kind,
            "monument_mode": self.monument_mode,
            "ignition_readiness": round(self.ignition_readiness, 4),
            "ignition_mode": self.ignition_mode,
            "activation_confidence": round(self.activation_confidence, 4),
            "recall_anchor": self.recall_anchor,
            "consolidation_pull": round(self.consolidation_pull, 4),
            "replay_priority": round(self.replay_priority, 4),
            "reconsolidation_priority": round(self.reconsolidation_priority, 4),
            "autobiographical_pull": round(self.autobiographical_pull, 4),
            "reconsolidation_mode": self.reconsolidation_mode,
            "forgetting_pressure": round(self.forgetting_pressure, 4),
            "memory_tension": round(self.memory_tension, 4),
            "prospective_pull": round(self.prospective_pull, 4),
            "dominant_relation_type": self.dominant_relation_type,
            "relation_generation_mode": self.relation_generation_mode,
            "relation_edges": [edge.to_dict() for edge in self.relation_edges],
            "meta_relations": [item.to_dict() for item in self.meta_relations],
            "dominant_causal_type": self.dominant_causal_type,
            "causal_generation_mode": self.causal_generation_mode,
            "causal_edges": [edge.to_dict() for edge in self.causal_edges],
            "dominant_mode": self.dominant_mode,
            "trace": [frame.to_dict() for frame in self.trace],
        }

    def to_packet_axes(
        self,
        previous: Mapping[str, Any] | "MemoryDynamicsState" | None = None,
    ) -> dict[str, dict[str, float]]:
        previous_state = coerce_memory_dynamics_state(previous)
        current_axes = _packet_axis_values(self)
        previous_axes = _packet_axis_values(previous_state)
        return {
            axis_name: {
                "value": round(axis_value, 4),
                "delta": round(axis_value - previous_axes.get(axis_name, 0.0), 4),
            }
            for axis_name, axis_value in current_axes.items()
        }


def derive_memory_dynamics_state(
    *,
    previous_state: Mapping[str, Any] | MemoryDynamicsState | None = None,
    memory_orchestration: Mapping[str, Any] | None = None,
    association_graph: Mapping[str, Any] | None = None,
    forgetting_snapshot: Mapping[str, Any] | None = None,
    sleep_consolidation: Mapping[str, Any] | None = None,
    activation_trace: Mapping[str, Any] | None = None,
    memory_palace_state: Mapping[str, Any] | None = None,
    recall_payload: Mapping[str, Any] | None = None,
    recall_active: bool = False,
    trace_limit: int = 8,
) -> MemoryDynamicsState:
    previous = coerce_memory_dynamics_state(previous_state)
    orchestration = _mapping(memory_orchestration)
    association = _association_payload(association_graph)
    forgetting = _mapping(forgetting_snapshot)
    sleep = _mapping(sleep_consolidation)
    activation = _activation_payload(activation_trace)
    palace = _memory_palace_payload(memory_palace_state)
    recall = _mapping(recall_payload)

    if (
        not orchestration
        and not association
        and not forgetting
        and not sleep
        and not activation
        and not palace
        and not recall
    ):
        return previous

    base_monument_salience = _float01(
        orchestration.get("monument_salience"),
        previous.monument_salience,
    )
    monument_kind = _text(orchestration.get("monument_kind")) or previous.monument_kind
    conscious_density = _float01(orchestration.get("conscious_mosaic_density"))
    conscious_recentness = _float01(orchestration.get("conscious_mosaic_recentness"))
    reuse_trajectory = _float01(orchestration.get("reuse_trajectory"))
    interference_pressure = _float01(orchestration.get("interference_pressure"))
    consolidation_priority = _float01(orchestration.get("consolidation_priority"))
    prospective_pull = _float01(
        orchestration.get("prospective_memory_pull"),
        previous.prospective_pull,
    )
    forgetting_pressure = _float01(
        forgetting.get("forgetting_pressure"),
        previous.forgetting_pressure,
    )
    dominant_link_key = _text(
        association.get("dominant_link_key")
        or association.get("dominant_link_id")
        or previous.dominant_link_key
    )
    dominant_link_inputs = tuple(
        _text(item)
        for item in association.get("dominant_inputs") or previous.dominant_link_inputs
        if _text(item)
    )
    dominant_weight = _float01(
        association.get("dominant_weight"),
        previous.palace_topology,
    )
    winner_margin = _float01(association.get("winner_margin"))

    replay_priority = _float01(
        sleep.get("replay_priority"),
        previous.replay_priority,
    )
    reconsolidation_priority = _float01(
        sleep.get("reconsolidation_priority"),
        previous.reconsolidation_priority,
    )
    autobiographical_pull = _float01(
        sleep.get("autobiographical_pull"),
        previous.autobiographical_pull,
    )

    palace_density_input = _clamp01(
        palace.get("topology_hint", 0.0) * 0.58
        + palace.get("active_density", 0.0) * 0.26
        + conscious_density * 0.08
        + conscious_recentness * 0.08
    )
    palace_density = _carry(
        previous.palace_density,
        palace_density_input,
        previous_state,
        0.18,
    )

    palace_topology = _clamp01(
        dominant_weight * 0.44
        + winner_margin * 0.12
        + conscious_density * 0.1
        + conscious_recentness * 0.1
        + reuse_trajectory * 0.08
        + palace_density * 0.16
    )
    if palace.get("topology_hint", 0.0) > 0.0:
        palace_topology = _clamp01(
            palace_topology * 0.84 + palace.get("topology_hint", 0.0) * 0.16
        )

    monument_salience = base_monument_salience
    if monument_salience <= 0.0 and palace.get("dominant_load", 0.0) > 0.0:
        monument_salience = _clamp01(palace.get("dominant_load", 0.0) * 0.72)
    if monument_salience <= 0.0 and activation.get("anchor_confirm", 0.0) > 0.0:
        monument_salience = _clamp01(activation.get("anchor_confirm", 0.0) * 0.72)

    activation_confidence = _clamp01(
        activation.get("internal_confidence", 0.0) * 0.68
        + activation.get("external_confidence", 0.0) * 0.32
    )
    if activation_confidence <= 0.0:
        activation_confidence = previous.activation_confidence

    ignition_readiness = _clamp01(
        monument_salience * 0.22
        + palace_topology * 0.16
        + reuse_trajectory * 0.16
        + prospective_pull * 0.14
        + conscious_recentness * 0.1
        + palace_density * 0.06
        + activation.get("chain_strength", 0.0) * 0.08
        + activation_confidence * 0.06
        + (0.12 if recall_active else 0.0)
        - interference_pressure * 0.12
        - forgetting_pressure * 0.1
    )

    consolidation_pull = _clamp01(
        consolidation_priority * 0.28
        + reconsolidation_priority * 0.22
        + replay_priority * 0.12
        + autobiographical_pull * 0.12
        + monument_salience * 0.08
        + palace_density * 0.06
        + max(0.0, 1.0 - interference_pressure) * 0.06
        + activation_confidence * 0.06
    )
    memory_tension = _clamp01(
        interference_pressure * 0.4
        + forgetting_pressure * 0.28
        + max(0.0, 1.0 - palace_topology) * 0.1
        + max(0.0, 1.0 - conscious_recentness) * 0.08
        + max(0.0, 1.0 - monument_salience) * 0.08
        + max(0.0, 1.0 - palace_density) * 0.06
    )

    recall_anchor = (
        _text(activation.get("anchor_hit"))
        or _text(recall.get("memory_anchor"))
        or previous.recall_anchor
    )

    palace_mode = _derive_palace_mode(
        palace_topology=palace_topology,
        palace_density=palace_density,
        dominant_link_key=dominant_link_key,
        winner_margin=winner_margin,
    )
    monument_mode = _derive_monument_mode(
        monument_salience=monument_salience,
        monument_kind=monument_kind,
    )
    ignition_mode = _derive_ignition_mode(
        ignition_readiness=ignition_readiness,
        activation_confidence=activation_confidence,
        chain_strength=activation.get("chain_strength", 0.0),
        recall_active=recall_active,
    )
    reconsolidation_mode = _derive_reconsolidation_mode(
        consolidation_pull=consolidation_pull,
        replay_priority=replay_priority,
        reconsolidation_priority=reconsolidation_priority,
        autobiographical_pull=autobiographical_pull,
        forgetting_pressure=forgetting_pressure,
    )
    relation_edges = _derive_relation_edges(
        association=association,
        dominant_link_key=dominant_link_key,
        dominant_link_inputs=dominant_link_inputs,
        recall_anchor=recall_anchor,
        winner_margin=winner_margin,
    )
    meta_relations = _derive_meta_relations(relation_edges)
    dominant_relation_type = relation_edges[0].relation_type if relation_edges else ""
    relation_generation_mode = _derive_relation_generation_mode(
        relation_edges=relation_edges,
        recall_active=recall_active,
        winner_margin=winner_margin,
    )
    causal_edges = _derive_causal_edges(
        relation_edges=relation_edges,
        meta_relations=meta_relations,
        activation=activation,
        recall_anchor=recall_anchor,
        memory_tension=memory_tension,
    )
    dominant_causal_type = causal_edges[0].causal_type if causal_edges else ""
    causal_generation_mode = _derive_causal_generation_mode(
        causal_edges=causal_edges,
        recall_active=recall_active,
        memory_tension=memory_tension,
    )
    dominant_mode = _derive_mode(
        ignition_readiness=ignition_readiness,
        ignition_mode=ignition_mode,
        consolidation_pull=consolidation_pull,
        reconsolidation_mode=reconsolidation_mode,
        memory_tension=memory_tension,
        prospective_pull=prospective_pull,
        recall_active=recall_active,
    )

    step = previous.trace[-1].step + 1 if previous.trace else 1
    trace = list(previous.trace[-max(0, trace_limit - 1) :]) if trace_limit > 0 else []
    trace.append(
        MemoryDynamicsFrame(
            step=step,
            palace_mode=palace_mode,
            monument_mode=monument_mode,
            ignition_mode=ignition_mode,
            reconsolidation_mode=reconsolidation_mode,
            recall_anchor=recall_anchor,
            palace_topology=palace_topology,
            palace_density=palace_density,
            monument_salience=monument_salience,
            ignition_readiness=ignition_readiness,
            activation_confidence=activation_confidence,
            consolidation_pull=consolidation_pull,
            memory_tension=memory_tension,
            prospective_pull=prospective_pull,
            dominant_mode=dominant_mode,
        )
    )

    return MemoryDynamicsState(
        palace_topology=palace_topology,
        palace_density=palace_density,
        palace_mode=palace_mode,
        dominant_link_key=dominant_link_key,
        dominant_link_inputs=dominant_link_inputs,
        monument_salience=monument_salience,
        monument_kind=monument_kind,
        monument_mode=monument_mode,
        ignition_readiness=ignition_readiness,
        ignition_mode=ignition_mode,
        activation_confidence=activation_confidence,
        recall_anchor=recall_anchor,
        consolidation_pull=consolidation_pull,
        replay_priority=replay_priority,
        reconsolidation_priority=reconsolidation_priority,
        autobiographical_pull=autobiographical_pull,
        reconsolidation_mode=reconsolidation_mode,
        forgetting_pressure=forgetting_pressure,
        memory_tension=memory_tension,
        prospective_pull=prospective_pull,
        dominant_relation_type=dominant_relation_type,
        relation_generation_mode=relation_generation_mode,
        relation_edges=relation_edges,
        meta_relations=meta_relations,
        dominant_causal_type=dominant_causal_type,
        causal_generation_mode=causal_generation_mode,
        causal_edges=causal_edges,
        dominant_mode=dominant_mode,
        trace=tuple(trace[-trace_limit:] if trace_limit > 0 else ()),
    )


def coerce_memory_dynamics_state(
    value: Mapping[str, Any] | MemoryDynamicsState | None,
) -> MemoryDynamicsState:
    if isinstance(value, MemoryDynamicsState):
        return value
    payload = dict(value or {})
    trace_items: list[MemoryDynamicsFrame] = []
    relation_edges: list[MemoryRelationEdge] = []
    meta_relations: list[MemoryMetaRelation] = []
    causal_edges: list[MemoryCausalEdge] = []
    for item in payload.get("relation_edges") or ():
        if isinstance(item, MemoryRelationEdge):
            relation_edges.append(item)
        elif isinstance(item, Mapping):
            relation_edges.append(
                MemoryRelationEdge(
                    relation_id=_text(item.get("relation_id")),
                    relation_key=_text(item.get("relation_key")),
                    relation_type=_text(item.get("relation_type")) or "association_bridge",
                    source_key=_text(item.get("source_key")),
                    target_key=_text(item.get("target_key")),
                    weight=_float01(item.get("weight")),
                    confidence=_float01(item.get("confidence")),
                    reasons=tuple(_text(reason) for reason in item.get("reasons") or () if _text(reason)),
                )
            )
    for item in payload.get("meta_relations") or ():
        if isinstance(item, MemoryMetaRelation):
            meta_relations.append(item)
        elif isinstance(item, Mapping):
            meta_relations.append(
                MemoryMetaRelation(
                    left_relation_id=_text(item.get("left_relation_id")),
                    right_relation_id=_text(item.get("right_relation_id")),
                    meta_type=_text(item.get("meta_type")) or "reinforces",
                    strength=_float01(item.get("strength")),
                )
            )
    for item in payload.get("causal_edges") or ():
        if isinstance(item, MemoryCausalEdge):
            causal_edges.append(item)
        elif isinstance(item, Mapping):
            causal_edges.append(
                MemoryCausalEdge(
                    causal_id=_text(item.get("causal_id")),
                    causal_key=_text(item.get("causal_key")),
                    causal_type=_text(item.get("causal_type")) or "enabled_by",
                    cause_key=_text(item.get("cause_key")),
                    effect_key=_text(item.get("effect_key")),
                    weight=_float01(item.get("weight")),
                    confidence=_float01(item.get("confidence")),
                    reasons=tuple(_text(reason) for reason in item.get("reasons") or () if _text(reason)),
                )
            )
    for item in payload.get("trace") or ():
        if isinstance(item, MemoryDynamicsFrame):
            trace_items.append(item)
            continue
        if not isinstance(item, Mapping):
            continue
        trace_items.append(
            MemoryDynamicsFrame(
                step=int(_float(item.get("step"), 0.0)),
                palace_mode=_text(item.get("palace_mode")) or "ambient",
                monument_mode=_text(item.get("monument_mode")) or "ambient",
                ignition_mode=_text(item.get("ignition_mode")) or "idle",
                reconsolidation_mode=_text(item.get("reconsolidation_mode")) or "settle",
                recall_anchor=_text(item.get("recall_anchor")),
                palace_topology=_float01(item.get("palace_topology")),
                palace_density=_float01(item.get("palace_density")),
                monument_salience=_float01(item.get("monument_salience")),
                ignition_readiness=_float01(item.get("ignition_readiness")),
                activation_confidence=_float01(item.get("activation_confidence")),
                consolidation_pull=_float01(item.get("consolidation_pull")),
                memory_tension=_float01(item.get("memory_tension")),
                prospective_pull=_float01(item.get("prospective_pull")),
                dominant_mode=_text(item.get("dominant_mode")) or "stabilize",
            )
        )
    return MemoryDynamicsState(
        palace_topology=_float01(payload.get("palace_topology")),
        palace_density=_float01(payload.get("palace_density")),
        palace_mode=_text(payload.get("palace_mode")) or "ambient",
        dominant_link_key=_text(payload.get("dominant_link_key")),
        dominant_link_inputs=tuple(
            _text(item)
            for item in payload.get("dominant_link_inputs") or ()
            if _text(item)
        ),
        monument_salience=_float01(payload.get("monument_salience")),
        monument_kind=_text(payload.get("monument_kind")),
        monument_mode=_text(payload.get("monument_mode")) or "ambient",
        ignition_readiness=_float01(payload.get("ignition_readiness")),
        ignition_mode=_text(payload.get("ignition_mode")) or "idle",
        activation_confidence=_float01(payload.get("activation_confidence")),
        recall_anchor=_text(payload.get("recall_anchor")),
        consolidation_pull=_float01(payload.get("consolidation_pull")),
        replay_priority=_float01(payload.get("replay_priority")),
        reconsolidation_priority=_float01(payload.get("reconsolidation_priority")),
        autobiographical_pull=_float01(payload.get("autobiographical_pull")),
        reconsolidation_mode=_text(payload.get("reconsolidation_mode")) or "settle",
        forgetting_pressure=_float01(payload.get("forgetting_pressure")),
        memory_tension=_float01(payload.get("memory_tension")),
        prospective_pull=_float01(payload.get("prospective_pull")),
        dominant_relation_type=_text(payload.get("dominant_relation_type")),
        relation_generation_mode=_text(payload.get("relation_generation_mode")) or "ambient",
        relation_edges=tuple(relation_edges),
        meta_relations=tuple(meta_relations),
        dominant_causal_type=_text(payload.get("dominant_causal_type")),
        causal_generation_mode=_text(payload.get("causal_generation_mode")) or "ambient",
        causal_edges=tuple(causal_edges),
        dominant_mode=_text(payload.get("dominant_mode")) or "stabilize",
        trace=tuple(trace_items),
    )


def _packet_axis_values(state: MemoryDynamicsState) -> dict[str, float]:
    return {
        "topology": _clamp01(state.palace_topology),
        "salience": _clamp01(state.monument_salience),
        "ignition": _clamp01(state.ignition_readiness),
        "consolidation": _clamp01(state.consolidation_pull),
        "tension": _clamp01(state.memory_tension),
        "replay": _clamp01(state.replay_priority),
        "reconsolidation": _clamp01(state.reconsolidation_priority),
        "confidence": _clamp01(state.activation_confidence),
    }


def _association_payload(value: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = _mapping(value)
    state_hint = payload.get("state_hint")
    if isinstance(state_hint, Mapping):
        merged = dict(state_hint)
        merged.setdefault("dominant_link_id", payload.get("dominant_link_id"))
        merged.setdefault("dominant_weight", payload.get("dominant_weight"))
        merged.setdefault("winner_margin", payload.get("winner_margin"))
        merged.setdefault("dominant_inputs", payload.get("dominant_inputs"))
        merged.setdefault("edges", payload.get("edges"))
        return merged
    return payload


def _derive_relation_edges(
    *,
    association: Mapping[str, Any],
    dominant_link_key: str,
    dominant_link_inputs: tuple[str, ...],
    recall_anchor: str,
    winner_margin: float,
) -> tuple[MemoryRelationEdge, ...]:
    raw_edges = association.get("edges")
    candidate_edges: list[MemoryRelationEdge] = []
    if isinstance(raw_edges, list):
        for item in raw_edges:
            if not isinstance(item, Mapping):
                continue
            relation_key = _text(item.get("link_key"))
            source_key = _text(item.get("left_seed_key"))
            target_key = _text(item.get("right_seed_key"))
            weight = _float01(item.get("weight"))
            reasons = tuple(_text(reason) for reason in item.get("reasons") or () if _text(reason))
            if not relation_key or not source_key or not target_key or weight <= 0.0:
                continue
            candidate_edges.append(
                MemoryRelationEdge(
                    relation_id=_text(item.get("link_id")) or relation_key,
                    relation_key=relation_key,
                    relation_type=_derive_relation_type(
                        reasons=reasons,
                        dominant_inputs=dominant_link_inputs,
                    ),
                    source_key=source_key,
                    target_key=target_key,
                    weight=weight,
                    confidence=_clamp01(weight * 0.76 + winner_margin * 0.24),
                    reasons=reasons,
                )
            )
    if candidate_edges:
        return tuple(sorted(candidate_edges, key=lambda item: item.weight, reverse=True)[:4])

    if not dominant_link_key:
        return ()
    left_key, right_key = _split_link_key(dominant_link_key)
    return (
        MemoryRelationEdge(
            relation_id=dominant_link_key or recall_anchor or "memory-relation",
            relation_key=dominant_link_key or recall_anchor or "memory-relation",
            relation_type=_derive_relation_type(
                reasons=dominant_link_inputs,
                dominant_inputs=dominant_link_inputs,
            ),
            source_key=left_key,
            target_key=right_key,
            weight=_float01(association.get("dominant_weight")),
            confidence=_clamp01(_float01(association.get("dominant_weight")) * 0.72 + winner_margin * 0.28),
            reasons=dominant_link_inputs,
        ),
    )


def _derive_meta_relations(
    relation_edges: tuple[MemoryRelationEdge, ...],
) -> tuple[MemoryMetaRelation, ...]:
    if len(relation_edges) < 2:
        return ()
    dominant = relation_edges[0]
    derived: list[MemoryMetaRelation] = []
    for edge in relation_edges[1:]:
        if edge.relation_type == dominant.relation_type:
            meta_type = "reinforces"
        elif {edge.source_key, edge.target_key} & {dominant.source_key, dominant.target_key}:
            meta_type = "specializes"
        else:
            meta_type = "competes_with"
        strength = _clamp01(min(dominant.weight, edge.weight) * 0.78 + min(dominant.confidence, edge.confidence) * 0.22)
        derived.append(
            MemoryMetaRelation(
                left_relation_id=dominant.relation_id,
                right_relation_id=edge.relation_id,
                meta_type=meta_type,
                strength=strength,
            )
        )
    return tuple(derived[:4])


def _derive_relation_generation_mode(
    *,
    relation_edges: tuple[MemoryRelationEdge, ...],
    recall_active: bool,
    winner_margin: float,
) -> str:
    if recall_active and relation_edges:
        return "ignited"
    if len(relation_edges) >= 2 and winner_margin <= 0.14:
        return "competitive"
    if len(relation_edges) >= 2:
        return "clustered"
    if relation_edges:
        return "anchored"
    return "ambient"


def _derive_causal_edges(
    *,
    relation_edges: tuple[MemoryRelationEdge, ...],
    meta_relations: tuple[MemoryMetaRelation, ...],
    activation: Mapping[str, Any],
    recall_anchor: str,
    memory_tension: float,
) -> tuple[MemoryCausalEdge, ...]:
    if not relation_edges:
        return ()
    meta_by_relation: dict[str, tuple[str, ...]] = {}
    for item in meta_relations:
        current = list(meta_by_relation.get(item.right_relation_id, ()))
        current.append(item.meta_type)
        meta_by_relation[item.right_relation_id] = tuple(current)
    activation_anchor = _text(activation.get("anchor_hit")) or recall_anchor
    derived: list[MemoryCausalEdge] = []
    for edge in relation_edges:
        meta_types = meta_by_relation.get(edge.relation_id, ())
        causal_type = _derive_causal_type(
            edge=edge,
            meta_types=meta_types,
            activation_anchor=activation_anchor,
            memory_tension=memory_tension,
        )
        cause_key = edge.source_key or activation_anchor or edge.relation_key
        effect_key = edge.target_key or edge.relation_key
        if recall_anchor and edge.relation_type in {"same_anchor", "unfinished_carry"}:
            cause_key = recall_anchor
        if not cause_key or not effect_key:
            continue
        reasons = tuple(
            item
            for item in [*edge.reasons, *meta_types, *( [f"anchor:{activation_anchor}"] if activation_anchor else [] )]
            if _text(item)
        )
        weight = _clamp01(
            edge.weight * 0.78
            + (0.12 if edge.relation_type == "same_anchor" and recall_anchor else 0.0)
            + (0.08 if "reinforces" in meta_types else 0.0)
            - (0.08 if causal_type == "suppressed_by" else 0.0)
        )
        confidence = _clamp01(
            edge.confidence * 0.74
            + (0.1 if activation_anchor else 0.0)
            + (0.08 if "reinforces" in meta_types else 0.0)
            - (0.06 if "competes_with" in meta_types and memory_tension >= 0.4 else 0.0)
        )
        derived.append(
            MemoryCausalEdge(
                causal_id=f"{edge.relation_id}:{causal_type}",
                causal_key=f"{cause_key}->{effect_key}:{causal_type}",
                causal_type=causal_type,
                cause_key=cause_key,
                effect_key=effect_key,
                weight=weight,
                confidence=confidence,
                reasons=reasons,
            )
        )
    return tuple(sorted(derived, key=lambda item: item.weight, reverse=True)[:4])


def _derive_causal_generation_mode(
    *,
    causal_edges: tuple[MemoryCausalEdge, ...],
    recall_active: bool,
    memory_tension: float,
) -> str:
    if recall_active and causal_edges:
        return "ignited"
    if any(edge.causal_type == "suppressed_by" for edge in causal_edges) and memory_tension >= 0.4:
        return "contested"
    if any(edge.causal_type == "amplified_by" for edge in causal_edges):
        return "reinforced"
    if causal_edges:
        return "anchored"
    return "ambient"


def _derive_causal_type(
    *,
    edge: MemoryRelationEdge,
    meta_types: tuple[str, ...],
    activation_anchor: str,
    memory_tension: float,
) -> str:
    if "competes_with" in meta_types and memory_tension >= 0.42:
        return "suppressed_by"
    if "reinforces" in meta_types and edge.relation_type in {"same_anchor", "recurrent_association", "association_bridge"}:
        return "amplified_by"
    if edge.relation_type == "unfinished_carry":
        return "reopened_by"
    if edge.relation_type == "cross_context_bridge":
        return "reframed_by"
    if edge.relation_type == "same_anchor":
        return "enabled_by"
    if edge.relation_type == "recurrent_association":
        return "triggered_by"
    if activation_anchor:
        return "triggered_by"
    return "enabled_by"


def _derive_relation_type(
    *,
    reasons: tuple[str, ...],
    dominant_inputs: tuple[str, ...],
) -> str:
    combined = {item for item in reasons if item} | {item for item in dominant_inputs if item}
    if "anchor_overlap" in combined:
        return "same_anchor"
    if "unfinished_thread" in combined or "unresolved_relief" in combined:
        return "unfinished_carry"
    if "source_diversity" in combined and "novelty_gain" in combined:
        return "cross_context_bridge"
    if "association_memory" in combined:
        return "recurrent_association"
    return "association_bridge"


def _split_link_key(value: str) -> tuple[str, str]:
    normalized = _text(value)
    if "->" in normalized:
        left, right = normalized.split("->", 1)
        return _text(left), _text(right)
    return normalized, ""


def _activation_payload(value: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = _mapping(value)
    if not payload:
        return {}
    chain = payload.get("activation_chain")
    chain_items = [item for item in chain if isinstance(item, Mapping)] if isinstance(chain, list) else []
    confidences = payload.get("confidence_curve")
    confidence_items = [item for item in confidences if isinstance(item, Mapping)] if isinstance(confidences, list) else []
    replay_events = payload.get("replay_events")
    replay_event_count = float(len(replay_events)) if isinstance(replay_events, list) else 0.0
    max_activation = 0.0
    mean_activation = 0.0
    if chain_items:
        activations = [_float01(item.get("activation")) for item in chain_items]
        max_activation = max(activations)
        mean_activation = sum(activations) / len(activations)
    internal_confidence = 0.0
    external_confidence = 0.0
    if confidence_items:
        final_sample = confidence_items[-1]
        internal_confidence = _float01(final_sample.get("conf_internal"))
        external_confidence = _float01(final_sample.get("conf_external"))
    chain_strength = _clamp01(max_activation * 0.62 + mean_activation * 0.38)
    anchor_confirm = _clamp01(
        chain_strength * 0.64
        + internal_confidence * 0.18
        + external_confidence * 0.08
        + min(replay_event_count, 3.0) / 3.0 * 0.1
    )
    return {
        "anchor_hit": _text(payload.get("anchor_hit")),
        "chain_strength": chain_strength,
        "internal_confidence": internal_confidence,
        "external_confidence": external_confidence,
        "replay_event_count": replay_event_count,
        "anchor_confirm": anchor_confirm,
    }


def _memory_palace_payload(value: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = _mapping(value)
    traces = payload.get("traces")
    if not isinstance(traces, Mapping):
        return {}
    nodes = payload.get("nodes")
    node_count = float(len(nodes)) if isinstance(nodes, list) else float(len(traces))
    if node_count <= 0.0:
        return {}

    active_nodes = 0.0
    dominant_node = ""
    dominant_load = 0.0
    total_load = 0.0
    for key, raw_trace in traces.items():
        if not isinstance(raw_trace, list):
            continue
        numeric_values = [
            _float(item, 0.0)
            for item in raw_trace
            if item is not None
        ]
        if not numeric_values:
            continue
        mean_load = _clamp01(sum(numeric_values) / len(numeric_values))
        total_load += mean_load
        if mean_load >= 0.08:
            active_nodes += 1.0
        if mean_load > dominant_load:
            dominant_load = mean_load
            dominant_node = _text(key)

    qualia_state = payload.get("qualia_state")
    qualia_memory = 0.0
    if isinstance(qualia_state, Mapping) and qualia_state:
        memory_values = [
            _float01(item.get("memory"))
            for item in qualia_state.values()
            if isinstance(item, Mapping)
        ]
        if memory_values:
            qualia_memory = sum(memory_values) / len(memory_values)

    active_density = _clamp01(active_nodes / node_count)
    mean_load = _clamp01(total_load / max(1.0, node_count))
    topology_hint = _clamp01(
        active_density * 0.46 + dominant_load * 0.34 + qualia_memory * 0.2
    )
    return {
        "active_density": active_density,
        "dominant_node": dominant_node,
        "dominant_load": dominant_load,
        "mean_load": mean_load,
        "qualia_memory": qualia_memory,
        "topology_hint": topology_hint,
    }


def _derive_palace_mode(
    *,
    palace_topology: float,
    palace_density: float,
    dominant_link_key: str,
    winner_margin: float,
) -> str:
    if palace_density >= 0.46 and winner_margin >= 0.16:
        return "clustered"
    if dominant_link_key and palace_topology >= 0.42:
        return "anchored"
    if palace_topology <= 0.18:
        return "sparse"
    if palace_density <= 0.22:
        return "diffuse"
    return "ambient"


def _derive_monument_mode(
    *,
    monument_salience: float,
    monument_kind: str,
) -> str:
    if monument_salience >= 0.62 and monument_kind:
        return "engraved"
    if monument_salience >= 0.42:
        return "rising"
    if monument_kind:
        return "tagged"
    return "ambient"


def _derive_ignition_mode(
    *,
    ignition_readiness: float,
    activation_confidence: float,
    chain_strength: float,
    recall_active: bool,
) -> str:
    if recall_active or chain_strength >= 0.52 or ignition_readiness >= 0.58:
        return "active"
    if ignition_readiness >= 0.4 or activation_confidence >= 0.42:
        return "primed"
    if ignition_readiness >= 0.24:
        return "arming"
    return "idle"


def _derive_reconsolidation_mode(
    *,
    consolidation_pull: float,
    replay_priority: float,
    reconsolidation_priority: float,
    autobiographical_pull: float,
    forgetting_pressure: float,
) -> str:
    if reconsolidation_priority >= max(replay_priority, autobiographical_pull, 0.52):
        return "reconsolidating"
    if replay_priority >= max(reconsolidation_priority, autobiographical_pull, 0.46):
        return "replaying"
    if forgetting_pressure >= 0.56 and consolidation_pull < 0.38:
        return "defragmenting"
    return "settle"


def _derive_mode(
    *,
    ignition_readiness: float,
    ignition_mode: str,
    consolidation_pull: float,
    reconsolidation_mode: str,
    memory_tension: float,
    prospective_pull: float,
    recall_active: bool,
) -> str:
    if recall_active or ignition_mode == "active":
        return "ignite"
    if reconsolidation_mode in {"reconsolidating", "replaying"} and consolidation_pull >= max(
        memory_tension,
        prospective_pull,
    ):
        return "reconsolidate"
    if prospective_pull >= max(ignition_readiness, consolidation_pull, memory_tension) and prospective_pull >= 0.42:
        return "prospect"
    if memory_tension >= 0.56 or reconsolidation_mode == "defragmenting":
        return "protect"
    return "stabilize"


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        candidate = value.to_dict()
        if isinstance(candidate, Mapping):
            return dict(candidate)
    return {}


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _float01(value: Any, default: float = 0.0) -> float:
    return _clamp01(_float(value, default))


def _carry(previous_value: float, current_value: float, previous_state: Any, alpha: float) -> float:
    if previous_state is None:
        return _clamp01(current_value)
    return _clamp01(previous_value * alpha + current_value * (1.0 - alpha))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
