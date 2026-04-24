from inner_os.memory_dynamics import (
    MemoryCausalEdge,
    MemoryDynamicsFrame,
    MemoryMetaRelation,
    MemoryRelationEdge,
    MemoryDynamicsState,
    coerce_memory_dynamics_state,
    derive_memory_dynamics_state,
)


def test_memory_dynamics_state_exposes_packet_axes() -> None:
    state = MemoryDynamicsState(
        palace_topology=0.44,
        palace_density=0.38,
        monument_salience=0.61,
        ignition_readiness=0.53,
        activation_confidence=0.49,
        consolidation_pull=0.47,
        replay_priority=0.41,
        reconsolidation_priority=0.43,
        memory_tension=0.28,
    )

    axes = state.to_packet_axes(
        {
            "palace_topology": 0.31,
            "palace_density": 0.32,
            "monument_salience": 0.55,
            "ignition_readiness": 0.41,
            "activation_confidence": 0.34,
            "consolidation_pull": 0.4,
            "replay_priority": 0.36,
            "reconsolidation_priority": 0.37,
            "memory_tension": 0.33,
        }
    )

    assert axes["topology"]["value"] == 0.44
    assert axes["topology"]["delta"] == 0.13
    assert axes["salience"]["value"] == 0.61
    assert axes["ignition"]["delta"] == 0.12
    assert axes["consolidation"]["delta"] == 0.07
    assert axes["tension"]["delta"] == -0.05
    assert axes["replay"]["delta"] == 0.05
    assert axes["reconsolidation"]["delta"] == 0.06
    assert axes["confidence"]["delta"] == 0.15


def test_derive_memory_dynamics_state_integrates_palace_monument_and_ignition_fragments() -> None:
    state = derive_memory_dynamics_state(
        previous_state=None,
        memory_orchestration={
            "monument_salience": 0.72,
            "monument_kind": "shared_ritual",
            "conscious_mosaic_density": 0.54,
            "conscious_mosaic_recentness": 0.67,
            "reuse_trajectory": 0.63,
            "interference_pressure": 0.22,
            "consolidation_priority": 0.58,
            "prospective_memory_pull": 0.41,
        },
        association_graph={
            "edges": [
                {
                    "link_id": "harbor->promise:4",
                    "link_key": "harbor->promise",
                    "left_seed_key": "harbor",
                    "right_seed_key": "promise",
                    "weight": 0.64,
                    "reasons": ["anchor_overlap", "unresolved_relief", "association_memory"],
                },
                {
                    "link_id": "promise->wind:2",
                    "link_key": "promise->wind",
                    "left_seed_key": "promise",
                    "right_seed_key": "wind",
                    "weight": 0.52,
                    "reasons": ["source_diversity", "novelty_gain"],
                },
            ],
            "state_hint": {
                "dominant_link_key": "harbor->promise",
                "dominant_weight": 0.64,
                "winner_margin": 0.36,
                "dominant_inputs": ["repeated_links", "unfinished_thread"],
            }
        },
        forgetting_snapshot={"forgetting_pressure": 0.18},
        sleep_consolidation={
            "reconsolidation_priority": 0.49,
            "replay_priority": 0.44,
            "autobiographical_pull": 0.46,
        },
        activation_trace={
            "anchor_hit": "harbor promise",
            "activation_chain": [
                {"node_id": "seed-1", "activation": 0.66},
                {"node_id": "seed-2", "activation": 0.42},
            ],
            "confidence_curve": [
                {"step": 0, "conf_internal": 0.42, "conf_external": 0.1},
                {"step": 1, "conf_internal": 0.67, "conf_external": 0.22},
            ],
            "replay_events": [{"scene_id": "scene-1"}],
        },
        memory_palace_state={
            "nodes": [{"name": "harbor"}, {"name": "promise"}],
            "traces": {
                "harbor": [0.7, 0.6, 0.5],
                "promise": [0.5, 0.44, 0.38],
            },
            "qualia_state": {
                "harbor": {"memory": 0.58},
                "promise": {"memory": 0.52},
            },
        },
        recall_payload={"memory_anchor": "harbor promise"},
        recall_active=True,
    )

    assert state.monument_kind == "shared_ritual"
    assert state.dominant_link_key == "harbor->promise"
    assert state.dominant_link_inputs == ("repeated_links", "unfinished_thread")
    assert state.palace_topology > 0.0
    assert state.palace_density > 0.0
    assert state.palace_mode in {"clustered", "anchored", "ambient"}
    assert state.monument_salience == 0.72
    assert state.monument_mode in {"engraved", "rising", "tagged"}
    assert state.ignition_readiness > 0.0
    assert state.ignition_mode == "active"
    assert state.activation_confidence > 0.0
    assert state.recall_anchor == "harbor promise"
    assert state.consolidation_pull > 0.0
    assert state.replay_priority == 0.44
    assert state.reconsolidation_priority == 0.49
    assert state.autobiographical_pull == 0.46
    assert state.reconsolidation_mode in {"reconsolidating", "replaying", "settle"}
    assert state.memory_tension >= 0.0
    assert state.dominant_relation_type == "same_anchor"
    assert state.relation_generation_mode in {"ignited", "clustered"}
    assert isinstance(state.relation_edges[0], MemoryRelationEdge)
    assert state.relation_edges[0].relation_key == "harbor->promise"
    assert state.relation_edges[0].relation_type == "same_anchor"
    assert state.meta_relations
    assert isinstance(state.meta_relations[0], MemoryMetaRelation)
    assert state.dominant_causal_type in {"enabled_by", "amplified_by"}
    assert state.causal_generation_mode in {"ignited", "reinforced", "anchored"}
    assert isinstance(state.causal_edges[0], MemoryCausalEdge)
    assert state.causal_edges[0].cause_key == "harbor promise"
    assert state.causal_edges[0].effect_key == "promise"
    assert state.dominant_mode == "ignite"
    assert state.trace
    assert state.trace[-1].recall_anchor == "harbor promise"


def test_derive_memory_dynamics_state_prefers_protect_mode_under_high_tension() -> None:
    state = derive_memory_dynamics_state(
        previous_state=coerce_memory_dynamics_state(
            {
                "palace_topology": 0.24,
                "palace_density": 0.26,
                "monument_salience": 0.22,
                "ignition_readiness": 0.18,
                "consolidation_pull": 0.21,
                "memory_tension": 0.4,
            }
        ),
        memory_orchestration={
            "monument_salience": 0.18,
            "conscious_mosaic_recentness": 0.14,
            "reuse_trajectory": 0.11,
            "interference_pressure": 0.81,
            "consolidation_priority": 0.12,
            "prospective_memory_pull": 0.15,
        },
        association_graph={
            "state_hint": {
                "dominant_link_key": "crowded->noise",
                "dominant_weight": 0.16,
                "winner_margin": 0.08,
                "dominant_inputs": ["interference_pressure"],
            }
        },
        forgetting_snapshot={"forgetting_pressure": 0.73},
        sleep_consolidation={"reconsolidation_priority": 0.08},
        recall_active=False,
    )

    assert state.memory_tension >= 0.56
    assert state.dominant_mode == "protect"
    assert state.ignition_mode in {"idle", "arming", "primed"}
    assert state.reconsolidation_mode in {"defragmenting", "settle"}


def test_coerce_memory_dynamics_state_preserves_trace_and_submodes() -> None:
    state = coerce_memory_dynamics_state(
        {
            "palace_topology": 0.33,
            "palace_density": 0.29,
            "palace_mode": "anchored",
            "monument_salience": 0.48,
            "monument_kind": "shared_ritual",
            "monument_mode": "rising",
            "ignition_readiness": 0.38,
            "ignition_mode": "primed",
            "activation_confidence": 0.41,
            "recall_anchor": "harbor promise",
            "consolidation_pull": 0.46,
            "replay_priority": 0.4,
            "reconsolidation_priority": 0.44,
            "autobiographical_pull": 0.31,
            "reconsolidation_mode": "replaying",
            "memory_tension": 0.22,
            "dominant_relation_type": "same_anchor",
            "relation_generation_mode": "anchored",
            "relation_edges": [
                {
                    "relation_id": "harbor->promise",
                    "relation_key": "harbor->promise",
                    "relation_type": "same_anchor",
                    "source_key": "harbor",
                    "target_key": "promise",
                    "weight": 0.48,
                    "confidence": 0.44,
                    "reasons": ["anchor_overlap"],
                }
            ],
            "meta_relations": [
                {
                    "left_relation_id": "harbor->promise",
                    "right_relation_id": "promise->wind",
                    "meta_type": "reinforces",
                    "strength": 0.39,
                }
            ],
            "dominant_causal_type": "enabled_by",
            "causal_generation_mode": "anchored",
            "causal_edges": [
                {
                    "causal_id": "harbor->promise:enabled_by",
                    "causal_key": "harbor->promise:enabled_by",
                    "causal_type": "enabled_by",
                    "cause_key": "harbor",
                    "effect_key": "promise",
                    "weight": 0.46,
                    "confidence": 0.51,
                    "reasons": ["anchor_overlap"],
                }
            ],
            "trace": [
                {
                    "step": 1,
                    "palace_mode": "anchored",
                    "monument_mode": "rising",
                    "ignition_mode": "primed",
                    "reconsolidation_mode": "replaying",
                    "recall_anchor": "harbor promise",
                    "palace_topology": 0.33,
                    "palace_density": 0.29,
                    "monument_salience": 0.48,
                    "ignition_readiness": 0.38,
                    "activation_confidence": 0.41,
                    "consolidation_pull": 0.46,
                    "memory_tension": 0.22,
                    "prospective_pull": 0.18,
                    "dominant_mode": "reconsolidate",
                }
            ],
        }
    )

    assert state.palace_mode == "anchored"
    assert state.monument_mode == "rising"
    assert state.ignition_mode == "primed"
    assert state.reconsolidation_mode == "replaying"
    assert state.recall_anchor == "harbor promise"
    assert state.dominant_relation_type == "same_anchor"
    assert state.relation_generation_mode == "anchored"
    assert isinstance(state.relation_edges[0], MemoryRelationEdge)
    assert state.dominant_causal_type == "enabled_by"
    assert state.causal_generation_mode == "anchored"
    assert isinstance(state.causal_edges[0], MemoryCausalEdge)
    assert isinstance(state.meta_relations[0], MemoryMetaRelation)
    assert isinstance(state.trace[0], MemoryDynamicsFrame)
