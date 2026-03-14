from examples.run_inner_os_integration_example import run_demo_turn


def test_run_demo_turn_produces_full_inner_os_cycle() -> None:
    result = run_demo_turn(
        "harbor signboard feels familiar",
        {
            "voice_level": 0.42,
            "body_stress_index": 0.33,
            "place_id": "harbor_market",
            "visual_cue": "signboard and slope",
        },
    )
    assert result["pre"]["state"]["stress"] >= 0.0
    assert result["recall"]["ignition_hints"]["recall_active"] is True
    assert result["gate"]["allowed_surface_intensity"] > 0.0
    assert result["llm"]["reply_text"].startswith("[")
    assert result["post"]["audit_record"]["kind"] == "thin_audit"
