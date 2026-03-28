from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .hook_contracts import (
    MemoryRecallInput,
    PostTurnUpdateInput,
    PreTurnUpdateInput,
    ResponseGateInput,
)
from .integration_hooks import IntegrationHooks
from .schemas import (
    INNER_OS_MEMORY_EVIDENCE_BUNDLE_SCHEMA,
    INNER_OS_MEMORY_RECALL_RESULT_SCHEMA,
    INNER_OS_POST_TURN_RESULT_SCHEMA,
    INNER_OS_PRE_TURN_RESULT_SCHEMA,
    INNER_OS_RECALL_PAYLOAD_SCHEMA,
    INNER_OS_RESPONSE_GATE_RESULT_SCHEMA,
)


class InnerOSService:
    """Thin service wrapper for existing systems.

    This keeps the public boundary small: callers can pass plain mappings
    and receive plain dictionaries without importing runtime internals.
    """

    def __init__(self, hooks: Optional[IntegrationHooks] = None) -> None:
        self.hooks = hooks or IntegrationHooks()

    def pre_turn_update(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        req = PreTurnUpdateInput.from_mapping(payload)
        result = self.hooks.pre_turn_update(
            user_input=req.user_input,
            sensor_input=req.sensor_input,
            local_context=req.local_context,
            current_state=req.current_state,
            safety_bias=req.safety_bias,
        )
        return _with_schema(result.to_dict(), INNER_OS_PRE_TURN_RESULT_SCHEMA)

    def memory_recall(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        req = MemoryRecallInput.from_mapping(payload)
        result = self.hooks.memory_recall(
            text_cue=req.text_cue,
            visual_cue=req.visual_cue,
            world_cue=req.world_cue,
            current_state=req.current_state,
            retrieval_summary=req.retrieval_summary,
        )
        out = _with_schema(result.to_dict(), INNER_OS_MEMORY_RECALL_RESULT_SCHEMA)
        out.setdefault("recall_payload_schema", INNER_OS_RECALL_PAYLOAD_SCHEMA)
        out.setdefault("memory_evidence_bundle_schema", INNER_OS_MEMORY_EVIDENCE_BUNDLE_SCHEMA)
        return out

    def response_gate(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        req = ResponseGateInput.from_mapping(payload)
        result = self.hooks.response_gate(
            draft=req.draft,
            current_state=req.current_state,
            safety_signals=req.safety_signals,
        )
        return _with_schema(result.to_dict(), INNER_OS_RESPONSE_GATE_RESULT_SCHEMA)

    def post_turn_update(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        req = PostTurnUpdateInput.from_mapping(payload)
        result = self.hooks.post_turn_update(
            user_input=req.user_input,
            output=req.output,
            current_state=req.current_state,
            memory_write_candidates=req.memory_write_candidates,
            recall_payload=req.recall_payload,
            transferred_lessons=req.transferred_lessons,
        )
        return _with_schema(result.to_dict(), INNER_OS_POST_TURN_RESULT_SCHEMA)


def _with_schema(payload: Mapping[str, Any], schema: str) -> Dict[str, Any]:
    out = dict(payload)
    out.setdefault("schema", schema)
    return out
