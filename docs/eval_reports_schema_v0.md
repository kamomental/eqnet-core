# Eval reports schema v0 (frozen)

## Purpose
- Persist shadow/canary evaluation outcomes for proposals.
- Keep evidence append-only and linkable by `proposal_id`.

## Freeze rules
- `required`: missing keys break audit/replay readers.
- `optional`: additive and backward compatible for v0 readers.

## `eval_report.v0`

Required keys:
- `schema_version`: must be `"eval_report.v0"`
- `eval_report_id`: UUID string
- `timestamp_ms`: integer epoch milliseconds (UTC)
- `proposal_id`: linked proposal id
- `method`: evaluation method id (e.g. `replay_eval`)
- `verdict`: enum-like string (`PASS|FAIL|INCONCLUSIVE`)
- `metrics_before`: object
- `metrics_after`: object
- `source_week`: string (`YYYY-Www`)

Optional keys:
- `delta`: object (metric diffs)

Missing-data behavior (frozen):
- If required fields are unavailable, emit nothing.
- Do not include plaintext user content or secrets.

