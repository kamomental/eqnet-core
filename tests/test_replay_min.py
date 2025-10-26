from emot_terrain_lab.mind.replay import UnifiedReplay


def test_replay_runs():
    replay = UnifiedReplay()
    trace = replay.rollout(z0={"x": 0}, horizon=2, value_fn=lambda state: state.get("x", 0))
    assert len(trace) == 2
