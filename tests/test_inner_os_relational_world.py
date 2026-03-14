from inner_os.relational_world import RelationalWorldCore


def test_relational_world_absorbs_culture_and_objects() -> None:
    core = RelationalWorldCore()
    snapshot = core.absorb_context(
        {
            "culture_id": "coastal",
            "community_id": "harbor_collective",
            "social_role": "companion",
            "place_memory_anchor": "harbor slope",
            "nearby_objects": ["signboard", "lantern"],
        }
    )
    assert snapshot["culture_id"] == "coastal"
    assert snapshot["community_id"] == "harbor_collective"
    assert snapshot["place_memory_anchor"] == "harbor slope"
    assert snapshot["nearby_objects"] == ["signboard", "lantern"]
