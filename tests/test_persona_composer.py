from persona import compose_controls


def test_compose_controls_basic():
    result = compose_controls(
        culture_name="anime_2010s_slice",
        mode_name="caregiver",
        user_pref={"warmth": 0.2, "emoji_use": 0.3},
        alpha=0.4,
        beta=0.3,
        safety={"beta": 0.3},
        base_controls={"directness": 0.0, "pause_ms": 100, "temp_mul": 1.0},
    )
    assert "controls" in result.to_dict()
    controls = result.controls
    assert -0.6 <= controls["directness"] <= 0.6
    assert 0 <= controls["pause_ms"] <= 600
    assert "warmth_bias" in controls
