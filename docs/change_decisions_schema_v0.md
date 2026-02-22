# Change decisions schema v0 (frozen)

## Purpose
- Persist accept/reject/rollback facts for change proposals.
- Keep decision history append-only and auditable.

## Freeze rules
- `required`: missing keys break proposal-decision link and readers.
- `optional`: additive and backward compatible for v0 readers.

## `change_decision.v0`

Required keys:
- `schema_version`: must be `"change_decision.v0"`
- `decision_id`: UUID string
- `timestamp_ms`: integer epoch milliseconds (UTC)
- `proposal_id`: linked proposal id
- `decision`: enum-like string (`REJECT|ACCEPT_SHADOW|ACCEPT_CANARY|ACCEPT_ROLLOUT|ROLLBACK|LINK_EVAL_REPORT`)
- `actor`: enum-like string (`auto|human`)
- `reason`: short explanation without secrets
- `source_week`: string (`YYYY-Www`)

Missing-data behavior (frozen):
- If required inputs are missing, fail fast and emit nothing.
- Do not include plaintext sensitive values.

Optional keys:
- `linked_eval_report_id`: linked eval report id for `LINK_EVAL_REPORT`
- `linked_eval_verdict`: eval verdict snapshot
- `linked_manifest_id`: manifest id snapshot
