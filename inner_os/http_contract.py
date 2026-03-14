from __future__ import annotations

from typing import Any, Dict

from .schemas import (
    INNER_OS_HTTP_MANIFEST_SCHEMA,
    INNER_OS_MEMORY_RECALL_INPUT_SCHEMA,
    INNER_OS_MEMORY_RECALL_RESULT_SCHEMA,
    INNER_OS_POST_TURN_INPUT_SCHEMA,
    INNER_OS_POST_TURN_RESULT_SCHEMA,
    INNER_OS_PRE_TURN_INPUT_SCHEMA,
    INNER_OS_PRE_TURN_RESULT_SCHEMA,
    INNER_OS_RECALL_PAYLOAD_SCHEMA,
    INNER_OS_RESPONSE_GATE_INPUT_SCHEMA,
    INNER_OS_RESPONSE_GATE_RESULT_SCHEMA,
    memory_record_contract,
    recall_payload_contract,
)


def build_inner_os_manifest() -> Dict[str, Any]:
    return {
        "schema": INNER_OS_HTTP_MANIFEST_SCHEMA,
        "service": "inner_os",
        "version": "0.1.0",
        "hooks": {
            "pre_turn_update": {
                "path": "/inner-os/pre-turn-update",
                "method": "POST",
                "request_schema": INNER_OS_PRE_TURN_INPUT_SCHEMA,
                "response_schema": INNER_OS_PRE_TURN_RESULT_SCHEMA,
            },
            "memory_recall": {
                "path": "/inner-os/memory-recall",
                "method": "POST",
                "request_schema": INNER_OS_MEMORY_RECALL_INPUT_SCHEMA,
                "response_schema": INNER_OS_MEMORY_RECALL_RESULT_SCHEMA,
                "recall_payload_schema": INNER_OS_RECALL_PAYLOAD_SCHEMA,
            },
            "response_gate": {
                "path": "/inner-os/response-gate",
                "method": "POST",
                "request_schema": INNER_OS_RESPONSE_GATE_INPUT_SCHEMA,
                "response_schema": INNER_OS_RESPONSE_GATE_RESULT_SCHEMA,
            },
            "post_turn_update": {
                "path": "/inner-os/post-turn-update",
                "method": "POST",
                "request_schema": INNER_OS_POST_TURN_INPUT_SCHEMA,
                "response_schema": INNER_OS_POST_TURN_RESULT_SCHEMA,
            },
        },
        "contracts": {
            "memory_record": memory_record_contract(),
            "recall_payload": recall_payload_contract(),
        },
        "notes": {
            "interaction_primary": True,
            "audit_thin": True,
            "simulation_subordinate_to_reality": True,
            "surface_swappable": True,
        },
    }
