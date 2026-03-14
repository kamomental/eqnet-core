from __future__ import annotations

from typing import Any, Dict

INNER_OS_HTTP_MANIFEST_SCHEMA = "inner_os_http_manifest/v1"
INNER_OS_MEMORY_RECORD_SCHEMA = "inner_os_memory/v1"
INNER_OS_RECALL_PAYLOAD_SCHEMA = "inner_os_recall_payload/v1"
INNER_OS_PRE_TURN_INPUT_SCHEMA = "inner_os_pre_turn_input/v1"
INNER_OS_PRE_TURN_RESULT_SCHEMA = "inner_os_pre_turn_result/v1"
INNER_OS_MEMORY_RECALL_INPUT_SCHEMA = "inner_os_memory_recall_input/v1"
INNER_OS_MEMORY_RECALL_RESULT_SCHEMA = "inner_os_memory_recall_result/v1"
INNER_OS_RESPONSE_GATE_INPUT_SCHEMA = "inner_os_response_gate_input/v1"
INNER_OS_RESPONSE_GATE_RESULT_SCHEMA = "inner_os_response_gate_result/v1"
INNER_OS_POST_TURN_INPUT_SCHEMA = "inner_os_post_turn_input/v1"
INNER_OS_POST_TURN_RESULT_SCHEMA = "inner_os_post_turn_result/v1"


def memory_record_contract() -> Dict[str, Any]:
    return {
        "schema": INNER_OS_MEMORY_RECORD_SCHEMA,
        "required_fields": ["kind", "summary", "text", "memory_anchor", "provenance"],
    }


def recall_payload_contract() -> Dict[str, Any]:
    return {
        "schema": INNER_OS_RECALL_PAYLOAD_SCHEMA,
        "required_fields": ["memory_anchor", "summary", "record_kind", "record_provenance"],
        "optional_fields": [
            "text",
            "source_episode_id",
            "policy_hint",
            "culture_id",
            "community_id",
            "social_role",
            "kind_breakdown",
        ],
    }
