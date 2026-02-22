# Golden replay manifest schema v0 (frozen)

## Purpose
- Freeze shadow replay input sets for reproducible evaluation.
- Block eval execution when input identity is not verified.

## `golden_replay_manifest.v0`

Required keys:
- `schema_version`: must be `"golden_replay_manifest.v0"`
- `manifest_id`: stable identifier
- `created_at_utc`: UTC timestamp string
- `policy`: object
- `inputs`: list of input groups

Input group required keys:
- `kind`: e.g. `telemetry_jsonl`
- `name`: logical group name
- `paths`: list of telemetry-dir relative paths
- `sha256`: list of expected sha256 hex strings (same length as `paths`)

Recommended metadata keys:
- `generator_git_commit`: commit hash of generator script/context
- `pii_filter_version_or_hash`: version/hash for PII filtering policy

## Missing-data behavior (frozen)
- Manifest missing: reject shadow eval (fail-safe).
- `inputs` empty: reject shadow eval by default.
- Input path missing: reject shadow eval.
- SHA mismatch: reject shadow eval.
- On reject, write no eval/link records.
