from inner_os.hook_contracts import (
    MemoryRecallInput,
    PostTurnUpdateInput,
    PreTurnUpdateInput,
    ResponseGateInput,
)
from inner_os.schemas import (
    INNER_OS_MEMORY_RECALL_INPUT_SCHEMA,
    INNER_OS_POST_TURN_INPUT_SCHEMA,
    INNER_OS_PRE_TURN_INPUT_SCHEMA,
    INNER_OS_RESPONSE_GATE_INPUT_SCHEMA,
)


def test_pre_turn_update_input_roundtrip() -> None:
    payload = {
        "user_input": {"text": "?????"},
        "sensor_input": {"body_stress_index": 0.2},
        "local_context": {"last_gate_context": {"life_indicator": 0.5}},
        "current_state": {"current_energy": 0.8},
        "safety_bias": 0.3,
    }
    dto = PreTurnUpdateInput.from_mapping(payload)
    assert dto.to_dict()["user_input"]["text"] == "?????"
    assert dto.safety_bias == 0.3
    assert dto.schema == INNER_OS_PRE_TURN_INPUT_SCHEMA
    assert dto.to_dict()["schema"] == INNER_OS_PRE_TURN_INPUT_SCHEMA


def test_memory_recall_input_roundtrip() -> None:
    payload = {
        "text_cue": "??",
        "visual_cue": "??",
        "world_cue": "market",
        "current_state": {"stress": 0.2},
        "retrieval_summary": {"hits": [{"id": "1"}]},
    }
    dto = MemoryRecallInput.from_mapping(payload)
    assert dto.to_dict()["visual_cue"] == "??"
    assert dto.retrieval_summary["hits"][0]["id"] == "1"
    assert dto.schema == INNER_OS_MEMORY_RECALL_INPUT_SCHEMA


def test_response_gate_input_roundtrip() -> None:
    payload = {
        "draft": {"text": "???"},
        "current_state": {"route": "conscious"},
        "safety_signals": {"safety_bias": 0.1},
    }
    dto = ResponseGateInput.from_mapping(payload)
    assert dto.current_state["route"] == "conscious"
    assert dto.to_dict()["draft"]["text"] == "???"
    assert dto.schema == INNER_OS_RESPONSE_GATE_INPUT_SCHEMA


def test_post_turn_update_input_roundtrip() -> None:
    payload = {
        "user_input": {"text": "?????"},
        "output": {"reply_text": "??????"},
        "current_state": {"memory_anchor": "??"},
        "memory_write_candidates": [{"kind": "observed", "summary": "seed"}],
        "transferred_lessons": [{"kind": "transferred_learning", "summary": "pause first"}],
    }
    dto = PostTurnUpdateInput.from_mapping(payload)
    assert dto.output["reply_text"] == "??????"
    assert dto.memory_write_candidates[0]["summary"] == "seed"
    assert dto.transferred_lessons[0]["kind"] == "transferred_learning"
    assert dto.schema == INNER_OS_POST_TURN_INPUT_SCHEMA
