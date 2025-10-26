from emot_terrain_lab.mind.mood_gate import mood_controls


def test_mood_small_gain_clip():
    base = {
        "temp_mul": 1.0,
        "top_p_mul": 1.0,
        "pause_ms": 0,
        "directness": 0.0,
    }
    mood = {"v": 0.1, "a": 0.4, "u": 0.3}
    out = mood_controls(base, mood, heartiness=0.6, style="chat_support")
    assert 0.6 <= out["temp_mul"] <= 1.2
    assert out["pause_ms"] >= 0
