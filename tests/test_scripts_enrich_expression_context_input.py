from scripts.enrich_expression_context_input import enrich_records


def test_enrich_records_merges_default_scenario_and_existing_context() -> None:
    records = [
        {
            "id": "case-1",
            "scenario": "vent_low",
            "expression_context_state": {
                "body": {"stress": 0.9},
                "custom": {"axis": "kept"},
            },
        }
    ]
    profiles = {
        "default": {
            "culture": {"politeness_pressure": 0.3},
            "body": {"stress": 0.2, "recovery_need": 0.4},
        },
        "vent_low": {
            "body": {"recovery_need": 0.7},
            "safety": {"dialogue_permission": "boundary_only"},
        },
    }

    enriched = enrich_records(records, profiles)

    context = enriched[0]["expression_context_state"]
    assert context["culture"]["politeness_pressure"] == 0.3
    assert context["body"]["stress"] == 0.9
    assert context["body"]["recovery_need"] == 0.7
    assert context["safety"]["dialogue_permission"] == "boundary_only"
    assert context["custom"]["axis"] == "kept"
