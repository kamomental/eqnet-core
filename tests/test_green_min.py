from emot_terrain_lab.mind.green import green_response


def test_green_soft_increases_pause():
    response = green_response({"tone": "soft", "semantic_valence": 0.6}, culture_resonance=0.5)
    assert response.controls["pause_ms_add"] > 0
    assert response.delta_mood["v"] > 0


def test_green_culture_kernel_modulates_axis():
    response = green_response(
        {"tone": "sharp"},
        culture_resonance=1.0,
        culture_kernel={"v": 0.0, "a": 1.0},
    )
    assert abs(response.delta_mood["v"]) < 1e-6
    assert response.delta_mood["a"] > 0
