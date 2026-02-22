# MECPE schema v0 (frozen)

## Purpose
- Persist multimodal emotion and cause signals in an auditable format.
- Minimize plaintext storage and keep hashes for reproducibility and tamper checks.

## Freeze rules
- `required`: missing keys break nightly audit/join contracts.
- `optional`: can be added or omitted without breaking v0 readers.

## A. `mecpe_record.v0` (turn-level minimum record)

Required keys:
- `schema_version`: must be `"mecpe_record.v0"`
- `timestamp_ms`: integer epoch milliseconds
- `turn_id`: string
- `prompt_hash`: sha256 hex string
- `model.version`: string
- `text_hash`: sha256 hex string
- `audio_sha256`: sha256 hex string or empty string when unavailable
- `video_sha256`: sha256 hex string or empty string when unavailable

Optional keys:
- `stage1_emotion`: `{polarity, emotion_fine, confidence, rationale_hash}`
- `stage2_cause_pair`: `{cause_turn_id, confidence, rationale_hash}`
- `stage3_cause_span`: `{start_char, end_char, span_hash}`
- `evidence`: `{audio_ref, video_ref}`

## Missing-data behavior (frozen)
- Emit a record even when audio/video evidence is absent.
- Do not store rationale plaintext in v0 telemetry. Use hash fields if needed.
