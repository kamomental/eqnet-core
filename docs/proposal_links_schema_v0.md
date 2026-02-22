# Proposal links schema v0 (frozen)

## Purpose
- Persist immutable links between proposals and evaluation artifacts.
- Support multiple evaluations per proposal without mutating proposal rows.

## Freeze rules
- `required`: missing keys break link traversal and audit chains.
- `optional`: additive and backward compatible for v0 readers.

## `proposal_link.v0`

Required keys:
- `schema_version`: must be `"proposal_link.v0"`
- `link_id`: UUID string
- `timestamp_ms`: integer epoch milliseconds (UTC)
- `proposal_id`: linked proposal id
- `eval_report_id`: linked eval report id
- `link_type`: enum-like string (`shadow_eval|canary_eval|rollout_eval`)
- `source_week`: string (`YYYY-Www`)

Missing-data behavior (frozen):
- If required fields are unavailable, emit nothing.
- Keep only identifiers and aggregate metadata.

