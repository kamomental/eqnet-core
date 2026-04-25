from scripts.core_quickstart_demo import build_core_demo_result


def test_small_shared_moment_demo_stays_small_and_joined() -> None:
    result = build_core_demo_result(scenario_name="small_shared_moment")

    contract = result["reaction_contract"]
    assert contract["stance"] == "join"
    assert contract["scale"] == "small"
    assert contract["question_budget"] == 0
    assert contract["interpretation_budget"] == "none"
    assert result["evaluation"]["passed"] is True
    assert result["expected_contract"]["stance"] == "join"
    audit = result["quick_audit_projection"]
    assert audit["schema_version"] == "quick_audit_projection.v1"
    assert audit["route"] == "core_quickstart"
    assert audit["audit_axes"]["response_channel"] == "speak"
    assert audit["organism_state"]["social_mode"] == "near"
    assert audit["memory_dynamics_state"]["dominant_relation_type"] == "same_anchor"
    assert "surface_context_source_state" in audit


def test_guarded_uncertainty_demo_prefers_hold() -> None:
    result = build_core_demo_result(scenario_name="guarded_uncertainty")

    contract = result["reaction_contract"]
    assert contract["stance"] == "hold"
    assert contract["response_channel"] == "hold"
    assert contract["continuity_mode"] == "reopen"
    assert result["evaluation"]["passed"] is True
    assert result["expected_contract"]["timing_mode"] == "held_open"
    audit = result["quick_audit_projection"]
    assert audit["audit_axes"]["response_channel"] == "hold"
    assert audit["audit_axes"]["organism_protective_tension"] == 0.64
    assert audit["audit_axes"]["memory_tension"] == 0.52
