from __future__ import annotations

from typing import Any


def redact_trace_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove raw transcript-like blobs while keeping lightweight surrogates."""

    redacted = dict(payload)
    redact_keys = (
        "raw_text",
        "raw_transcript",
        "transcript",
        "messages",
        "prompt",
        "completion",
        "input_text",
        "output_text",
    )
    for key in redact_keys:
        if key not in redacted:
            continue
        value = redacted.pop(key)
        if isinstance(value, str):
            redacted[f"{key}_char_len"] = len(value)
        elif isinstance(value, list):
            redacted[f"{key}_items"] = len(value)
    return redacted


__all__ = ["redact_trace_payload"]
