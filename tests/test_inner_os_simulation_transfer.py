from inner_os.simulation_transfer import SimulationTransferCore


def test_simulation_transfer_core_promotes_abstract_lesson() -> None:
    core = SimulationTransferCore()
    promoted = core.promote(
        [
            {
                "episode_id": "sim-001",
                "summary": "pause, observe, then clarify before acting",
                "patterns": ["pause and observe when signals conflict"],
                "benefit_score": 0.8,
                "risk_score": 0.2,
                "transfer_ready": True,
            }
        ]
    )
    assert len(promoted) == 1
    assert promoted[0].policy_hint == "pause_and_observe_under_ambiguity"
    assert promoted[0].to_memory_record()["kind"] == "transferred_learning"


def test_simulation_transfer_core_blocks_contradictory_episode() -> None:
    core = SimulationTransferCore()
    promoted = core.promote(
        [
            {
                "episode_id": "sim-002",
                "summary": "this real person always rejects me",
                "patterns": ["assume rejection early"],
                "benefit_score": 0.9,
                "risk_score": 0.1,
                "contradiction_with_real": True,
                "transfer_ready": True,
            }
        ]
    )
    assert promoted == []
