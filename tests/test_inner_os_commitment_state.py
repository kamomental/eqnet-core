from inner_os.commitment_state import derive_commitment_state


def test_temporal_reentry_bias_lifts_repair_commitment_target() -> None:
    base_kwargs = {
        "qualia_planner_view": {
            "body_load": 0.08,
            "protection_bias": 0.14,
            "degraded": False,
        },
        "terrain_readout": {
            "approach_bias": 0.34,
            "protect_bias": 0.16,
        },
        "protection_mode": {
            "mode": "repair",
            "strength": 0.46,
            "winner_margin": 0.12,
        },
        "body_recovery_guard": {
            "state": "open",
            "score": 0.16,
        },
        "initiative_readiness": {
            "state": "tentative",
            "score": 0.42,
            "winner_margin": 0.08,
        },
        "initiative_followup_bias": {
            "state": "reopen_softly",
            "score": 0.44,
        },
        "temperament_estimate": {
            "leader_tendency": 0.54,
            "hero_tendency": 0.24,
            "bond_drive": 0.42,
            "recovery_discipline": 0.2,
            "protect_floor": 0.12,
            "risk_tolerance": 0.28,
        },
        "memory_write_class": "repair_trace",
        "memory_write_class_reason": "repair_trace",
        "insight_event": {
            "triggered": True,
            "orient_bias": 0.16,
        },
    }

    without_bias = derive_commitment_state(
        self_state={
            "stress": 0.18,
            "recovery_need": 0.14,
        },
        **base_kwargs,
    ).to_dict()
    with_bias = derive_commitment_state(
        self_state={
            "stress": 0.18,
            "recovery_need": 0.14,
            "temporal_reentry_pull": 0.62,
            "temporal_relation_reentry_pull": 0.54,
            "temporal_continuity_pressure": 0.48,
            "temporal_timeline_coherence": 0.44,
            "temporal_membrane_mode": "reentry",
        },
        **base_kwargs,
    ).to_dict()

    assert with_bias["target_scores"]["repair"] > without_bias["target_scores"]["repair"]
    assert with_bias["scores"]["commit"] >= without_bias["scores"]["commit"]
    assert "temporal_reentry_pull" in with_bias["dominant_inputs"]


def test_temporal_supersession_bias_lifts_hold_targets_and_waver() -> None:
    base_kwargs = {
        "qualia_planner_view": {
            "body_load": 0.06,
            "protection_bias": 0.1,
            "degraded": False,
        },
        "terrain_readout": {
            "approach_bias": 0.38,
            "protect_bias": 0.14,
        },
        "protection_mode": {
            "mode": "monitor",
            "strength": 0.22,
            "winner_margin": 0.1,
        },
        "body_recovery_guard": {
            "state": "open",
            "score": 0.12,
        },
        "initiative_readiness": {
            "state": "ready",
            "score": 0.56,
            "winner_margin": 0.14,
        },
        "initiative_followup_bias": {
            "state": "offer_next_step",
            "score": 0.38,
        },
        "temperament_estimate": {
            "leader_tendency": 0.22,
            "hero_tendency": 0.52,
            "bond_drive": 0.34,
            "recovery_discipline": 0.16,
            "protect_floor": 0.18,
            "risk_tolerance": 0.4,
        },
        "memory_write_class": "safe_repeat",
        "memory_write_class_reason": "safe_repeat",
        "insight_event": {
            "triggered": False,
            "orient_bias": 0.0,
        },
    }

    without_bias = derive_commitment_state(
        self_state={
            "stress": 0.14,
            "recovery_need": 0.12,
        },
        **base_kwargs,
    ).to_dict()
    with_bias = derive_commitment_state(
        self_state={
            "stress": 0.14,
            "recovery_need": 0.12,
            "temporal_supersession_pressure": 0.68,
            "temporal_timeline_coherence": 0.22,
            "temporal_membrane_mode": "supersede",
        },
        **base_kwargs,
    ).to_dict()

    assert with_bias["target_scores"]["hold"] > without_bias["target_scores"]["hold"]
    assert with_bias["scores"]["waver"] > without_bias["scores"]["waver"]
    assert any(
        item in with_bias["dominant_inputs"]
        for item in {"temporal_supersession_pressure", "temporal_timeline_coherence"}
    )


def test_temporal_sleep_bias_fields_work_as_weak_fallback() -> None:
    base_kwargs = {
        "qualia_planner_view": {
            "body_load": 0.08,
            "protection_bias": 0.14,
            "degraded": False,
        },
        "terrain_readout": {
            "approach_bias": 0.34,
            "protect_bias": 0.16,
        },
        "protection_mode": {
            "mode": "repair",
            "strength": 0.46,
            "winner_margin": 0.12,
        },
        "body_recovery_guard": {
            "state": "open",
            "score": 0.16,
        },
        "initiative_readiness": {
            "state": "tentative",
            "score": 0.42,
            "winner_margin": 0.08,
        },
        "initiative_followup_bias": {
            "state": "reopen_softly",
            "score": 0.44,
        },
        "temperament_estimate": {
            "leader_tendency": 0.54,
            "hero_tendency": 0.24,
            "bond_drive": 0.42,
            "recovery_discipline": 0.2,
            "protect_floor": 0.12,
            "risk_tolerance": 0.28,
        },
        "memory_write_class": "repair_trace",
        "memory_write_class_reason": "repair_trace",
        "insight_event": {
            "triggered": True,
            "orient_bias": 0.16,
        },
    }

    baseline = derive_commitment_state(
        self_state={"stress": 0.18, "recovery_need": 0.14},
        **base_kwargs,
    ).to_dict()
    carried = derive_commitment_state(
        self_state={
            "stress": 0.18,
            "recovery_need": 0.14,
            "temporal_membrane_focus": "reentry",
            "temporal_timeline_bias": 0.12,
            "temporal_reentry_bias": 0.17,
            "temporal_continuity_bias": 0.11,
            "temporal_relation_reentry_bias": 0.09,
        },
        **base_kwargs,
    ).to_dict()

    assert carried["target_scores"]["repair"] > baseline["target_scores"]["repair"]
    assert carried["scores"]["commit"] >= baseline["scores"]["commit"]
