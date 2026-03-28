from inner_os.schemas import (
    INNER_OS_DASHBOARD_SNAPSHOT_SCHEMA,
    INNER_OS_DISTILLATION_RECORD_SCHEMA,
    INNER_OS_HTTP_MANIFEST_SCHEMA,
    INNER_OS_MEMORY_EVIDENCE_BUNDLE_SCHEMA,
    INNER_OS_MEMORY_RECORD_SCHEMA,
    INNER_OS_RECALL_PAYLOAD_SCHEMA,
    INNER_OS_TRANSFER_PACKAGE_SCHEMA,
    INNER_OS_WORKING_MEMORY_SNAPSHOT_SCHEMA,
    dashboard_snapshot_contract,
    distillation_record_contract,
    memory_evidence_bundle_contract,
    memory_record_contract,
    recall_payload_contract,
    transfer_package_contract,
)


def test_schema_contract_helpers_expose_expected_names() -> None:
    assert INNER_OS_HTTP_MANIFEST_SCHEMA == "inner_os_http_manifest/v1"
    assert INNER_OS_DISTILLATION_RECORD_SCHEMA == "inner_os_distillation_record/v1"
    assert INNER_OS_TRANSFER_PACKAGE_SCHEMA == "inner_os_transfer_package/v1"
    assert INNER_OS_DASHBOARD_SNAPSHOT_SCHEMA == "inner_os_dashboard_snapshot/v1"
    assert INNER_OS_MEMORY_EVIDENCE_BUNDLE_SCHEMA == "inner_os_memory_evidence_bundle/v1"
    assert INNER_OS_MEMORY_RECORD_SCHEMA == "inner_os_memory/v1"
    assert INNER_OS_RECALL_PAYLOAD_SCHEMA == "inner_os_recall_payload/v1"
    assert INNER_OS_WORKING_MEMORY_SNAPSHOT_SCHEMA == "inner_os_working_memory_snapshot/v1"

    distill_contract = distillation_record_contract()
    transfer_contract = transfer_package_contract()
    dashboard_contract = dashboard_snapshot_contract()
    evidence_contract = memory_evidence_bundle_contract()
    memory_contract = memory_record_contract()
    recall_contract = recall_payload_contract()
    assert distill_contract["schema"] == INNER_OS_DISTILLATION_RECORD_SCHEMA
    assert transfer_contract["schema"] == INNER_OS_TRANSFER_PACKAGE_SCHEMA
    assert dashboard_contract["schema"] == INNER_OS_DASHBOARD_SNAPSHOT_SCHEMA
    assert evidence_contract["schema"] == INNER_OS_MEMORY_EVIDENCE_BUNDLE_SCHEMA
    assert memory_contract["schema"] == INNER_OS_MEMORY_RECORD_SCHEMA
    assert recall_contract["schema"] == INNER_OS_RECALL_PAYLOAD_SCHEMA
    assert "facts_current" in evidence_contract["required_fields"]
    assert "reentry_contexts" in evidence_contract["required_fields"]
    assert recall_contract["required_fields"] == ["memory_anchor", "summary", "record_kind", "record_provenance"]
    assert "terrain_observed_roughness" in recall_contract["optional_fields"]
    assert "terrain_transition_roughness" in recall_contract["optional_fields"]
    assert "decision_snapshot" in distill_contract["required_fields"]
    assert "portable_state" in transfer_contract["required_fields"]
    assert "same_turn" in dashboard_contract["required_fields"]
