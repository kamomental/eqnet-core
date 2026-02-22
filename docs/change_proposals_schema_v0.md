# Change proposals schema v0 (frozen)

## Purpose
- Persist suggested safe changes derived from audit outputs.
- Keep v0 append-only; no auto-apply.

## Freeze rules
- `required`: missing keys break readers/weekly pipelines.
- `optional`: additive only, backward compatible for v0 readers.

## `change_proposal.v0`

Required keys:
- `schema_version`: must be `"change_proposal.v0"`
- `proposal_id`: UUID string
- `timestamp_ms`: integer epoch milliseconds (UTC)
- `trigger`: object (source audit summary)
- `suggested_change`: object (what to change)
- `expected_effect`: object (expected KPI impact)
- `risk_level`: enum-like string (`LOW|MED|HIGH`)
- `requires_gate`: enum-like string (`shadow|canary|rollout`)
- `source_week`: string (`YYYY-Www`)

Missing-data behavior (frozen):
- If no suggestion is generated, emit nothing.
- Do not store plaintext sensitive values; keep identifiers or hashes.

