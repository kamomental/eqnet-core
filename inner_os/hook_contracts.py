from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Mapping

from .schemas import (
    INNER_OS_MEMORY_RECALL_INPUT_SCHEMA,
    INNER_OS_POST_TURN_INPUT_SCHEMA,
    INNER_OS_PRE_TURN_INPUT_SCHEMA,
    INNER_OS_RESPONSE_GATE_INPUT_SCHEMA,
)


@dataclass
class PreTurnUpdateInput:
    user_input: Dict[str, Any] = field(default_factory=dict)
    sensor_input: Dict[str, Any] = field(default_factory=dict)
    local_context: Dict[str, Any] = field(default_factory=dict)
    current_state: Dict[str, Any] = field(default_factory=dict)
    safety_bias: float = 0.0
    schema: str = INNER_OS_PRE_TURN_INPUT_SCHEMA

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PreTurnUpdateInput":
        return cls(
            user_input=dict(payload.get("user_input") or {}),
            sensor_input=dict(payload.get("sensor_input") or {}),
            local_context=dict(payload.get("local_context") or {}),
            current_state=dict(payload.get("current_state") or {}),
            safety_bias=float(payload.get("safety_bias") or 0.0),
            schema=str(payload.get("schema") or INNER_OS_PRE_TURN_INPUT_SCHEMA),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryRecallInput:
    text_cue: str = ""
    visual_cue: str = ""
    world_cue: str = ""
    current_state: Dict[str, Any] = field(default_factory=dict)
    retrieval_summary: Dict[str, Any] = field(default_factory=dict)
    schema: str = INNER_OS_MEMORY_RECALL_INPUT_SCHEMA

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MemoryRecallInput":
        return cls(
            text_cue=str(payload.get("text_cue") or ""),
            visual_cue=str(payload.get("visual_cue") or ""),
            world_cue=str(payload.get("world_cue") or ""),
            current_state=dict(payload.get("current_state") or {}),
            retrieval_summary=dict(payload.get("retrieval_summary") or {}),
            schema=str(payload.get("schema") or INNER_OS_MEMORY_RECALL_INPUT_SCHEMA),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResponseGateInput:
    draft: Dict[str, Any] = field(default_factory=dict)
    current_state: Dict[str, Any] = field(default_factory=dict)
    safety_signals: Dict[str, Any] = field(default_factory=dict)
    schema: str = INNER_OS_RESPONSE_GATE_INPUT_SCHEMA

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ResponseGateInput":
        return cls(
            draft=dict(payload.get("draft") or {}),
            current_state=dict(payload.get("current_state") or {}),
            safety_signals=dict(payload.get("safety_signals") or {}),
            schema=str(payload.get("schema") or INNER_OS_RESPONSE_GATE_INPUT_SCHEMA),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PostTurnUpdateInput:
    user_input: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    current_state: Dict[str, Any] = field(default_factory=dict)
    memory_write_candidates: list[Dict[str, Any]] = field(default_factory=list)
    recall_payload: Dict[str, Any] = field(default_factory=dict)
    transferred_lessons: list[Dict[str, Any]] = field(default_factory=list)
    schema: str = INNER_OS_POST_TURN_INPUT_SCHEMA

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PostTurnUpdateInput":
        raw_candidates = payload.get("memory_write_candidates") or []
        raw_lessons = payload.get("transferred_lessons") or []
        return cls(
            user_input=dict(payload.get("user_input") or {}),
            output=dict(payload.get("output") or {}),
            current_state=dict(payload.get("current_state") or {}),
            memory_write_candidates=[dict(item) for item in raw_candidates if isinstance(item, Mapping)],
            recall_payload=dict(payload.get("recall_payload") or {}),
            transferred_lessons=[dict(item) for item in raw_lessons if isinstance(item, Mapping)],
            schema=str(payload.get("schema") or INNER_OS_POST_TURN_INPUT_SCHEMA),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
