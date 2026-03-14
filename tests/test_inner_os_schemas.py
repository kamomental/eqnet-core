from inner_os.schemas import (
    INNER_OS_HTTP_MANIFEST_SCHEMA,
    INNER_OS_MEMORY_RECORD_SCHEMA,
    INNER_OS_RECALL_PAYLOAD_SCHEMA,
    memory_record_contract,
    recall_payload_contract,
)


def test_schema_contract_helpers_expose_expected_names() -> None:
    assert INNER_OS_HTTP_MANIFEST_SCHEMA == "inner_os_http_manifest/v1"
    assert INNER_OS_MEMORY_RECORD_SCHEMA == "inner_os_memory/v1"
    assert INNER_OS_RECALL_PAYLOAD_SCHEMA == "inner_os_recall_payload/v1"

    memory_contract = memory_record_contract()
    recall_contract = recall_payload_contract()
    assert memory_contract["schema"] == INNER_OS_MEMORY_RECORD_SCHEMA
    assert recall_contract["schema"] == INNER_OS_RECALL_PAYLOAD_SCHEMA
    assert recall_contract["required_fields"] == ["memory_anchor", "summary", "record_kind", "record_provenance"]
