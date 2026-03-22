from inner_os.memory_orchestration_core import MemoryOrchestrationCore


def test_memory_orchestration_core_builds_compact_state() -> None:
    core = MemoryOrchestrationCore(
        monument_query=lambda world: (0.72, "shared_ritual")
    )
    snapshot = core.snapshot(
        relational_world={
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
        },
        current_state={
            "conscious_mosaic_density": 0.5,
            "conscious_mosaic_recentness": 0.8,
            "replay_intensity": 0.44,
            "anticipation_tension": 0.36,
            "future_signal": 0.41,
            "terrain_transition_roughness": 0.23,
        },
        forgetting_snapshot={
            "forgetting_pressure": 0.28,
            "replay_horizon": 3,
        },
        recall_active=True,
    )
    data = snapshot.to_dict()
    assert data["monument_salience"] == 0.72
    assert data["monument_kind"] == "shared_ritual"
    assert data["reuse_trajectory"] > 0.0
    assert data["interference_pressure"] >= 0.0
    assert data["consolidation_priority"] > 0.0
    assert data["prospective_memory_pull"] > 0.0


def test_memory_orchestration_core_uses_partner_relation_bias() -> None:
    core = MemoryOrchestrationCore(monument_query=lambda world: (0.0, None))
    without_partner = core.snapshot(
        relational_world={},
        current_state={
            "attachment": 0.72,
            "familiarity": 0.68,
            "trust_memory": 0.7,
            "relation_seed_strength": 0.74,
        },
    )
    with_partner = core.snapshot(
        relational_world={},
        current_state={
            "related_person_id": "user",
            "attachment": 0.72,
            "familiarity": 0.68,
            "trust_memory": 0.7,
            "relation_seed_strength": 0.74,
        },
    )
    assert with_partner.reuse_trajectory > without_partner.reuse_trajectory
    assert with_partner.consolidation_priority > without_partner.consolidation_priority
    assert with_partner.prospective_memory_pull > without_partner.prospective_memory_pull
