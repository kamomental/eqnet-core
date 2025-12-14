Mapper Version: mapper_v2

﻿# Perception Contract

This document captures how raw observations are converted into the PerceptPacket that feeds the heart loop. It is the legal contract for "how the world is felt".

## SomaticSignals

| Field | Meaning | Range | Source / Conversion | Notes |
|-------|---------|-------|----------------------|-------|
| `arousal_hint` | instantaneous arousal / activation | 0-1 | direct from sensor metric `arousal_hint`; NaN/∞ -> default 0 | stored as float, extras keeps raw keys |
| `stress_hint` | perceived bodily stress | 0-1 | `stress_hint` \|\| `body_stress_index` \|\| `stress`; 0.0 preserved; `_first_not_none` used | extras contains `distance_raw` etc |
| `fatigue_hint` | tiredness / sleep debt | 0-1 | `fatigue_hint` \|\| `fatigue` \|\| `sleep_debt` | ... |
| `jitter` | motion jitter / instability | 0-1 | `jitter` \|\| `motion_score` | ... |
| `proximity` | closeness sense | 0-1 | direct `proximity` else normalized `distance` via `1/(1+d)`; `distance_raw` kept | 0.0 means overlapping, 1.0 far |
| `extras` | unnormalized raw fields | -- | everything not listed above | never include raw transcripts |

Quality: sensors may set extras like `metrics.pose_detected`, `body_state_flag`. Missing metrics leave hints at default 0.

## SocialContext

| Field | Meaning | Range | Conversion |
|-------|---------|-------|------------|
| `mode` | high-level context style | categorical string | `mode` or `style` default "casual" |
| `cultural_pressure` | social pressure | 0-1 | `_coerce_float(..., clamp01=True)` |
| `offer_requested` | whether advice was invited | boolean | `_coerce_bool(offer_requested)` or `request in {advice,help}` |
| `disclosure_budget` | self-disclosure allowance | 0-1 | `disclosure_budget` \|\| `intimacy`; raw `intimacy` stored in extras |

## WorldSummary

| Field | Meaning | Range | Conversion |
|-------|---------|-------|------------|
| `hazard_level` | external risk | 0-1 | `hazard_level` \|\| `danger` \|\| `threat` |
| `ambiguity` | uncertainty of scene | 0-1 | `ambiguity` \|\| `uncertainty`; if missing but `clarity` present -> `1 - clamp01(clarity)` |
| `npc_affect` | attitude of NPC | -1..1 | `npc_affect` \|\| `npc_valence` |
| `social_pressure` | crowd / stage pressure | 0-1 | `social_pressure` \|\| `crowd_pressure` |
| `extras` | geography/lighting references etc | -- | retains original keys including `clarity_raw` |

## Deterministic fields

When `--deterministic` flag is used (CI fixtures), we force:

- `scenario_id = <pair key>`
- `turn_id = f"{scenario_id}-turn-{idx:04d}`
- `timestamp_ms = idx * 1000`
- `seed = idx + 1`

This ensures fixtures do not drift and diff noise stays zero.

## Quality & PII rules

- Raw transcripts, names, IDs are never put inside PerceptPacket; use `<redacted>` or references.
- Raw sensor units (distance, clarity) are stored in `extras` with suffix `_raw` alongside normalization formula in `mapper` metadata.
- Observations recorded in trace include `observations.mapper.{version,...}`, `observations.<source>` where `<source>` is `hub` or `runtime`.

## Mapper metadata

Each trace writes `policy.observations.mapper` (and `qualia.observations.mapper`) with:

```json
{
  "version": "mapper_v2",
  "distance_norm": {"mode": "reciprocal", "formula": "1/(1+d)"},
  "clarity_inverted": true,
  "clamp_hints": true,
  "nan_policy": "reset_to_default"
}
```

This metadata is the audit trail proving how perception was constructed.

