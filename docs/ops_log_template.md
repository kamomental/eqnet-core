# Operational Log Template (HeartOS)

Purpose: record runs so decisions, constraints, and recovery can be explained and replayed.

## Assumptions

- One line per event (jsonl).
- One run is a coherent life slice (single trace root).

## Required Fields (Minimum)

- timestamp_ms
- seed
- world_type
- post_risk_scale
- transition_uncertainty_factor
- decision

## Recommended Fields (Context)

- run_id
- scenario_id
- step_index
- transition_turn_index
- boundary.score
- boundary.reasons
- risk / uncertainty / tom_cost / drive
- veto_score
- notes

## Optional Interpretation Fields

- decision_reason (short free text)
- override_context (param/from/to/reason)
- world_snapshot (lightweight tags for context)
- postmortem_note (added after review)

## Good vs Bad Log (Quick Check)

Good:
- run_id is present and trace_root is unique.
- decision and boundary are present on every decision_cycle.
- transition events are explicitly recorded.

Bad:
- mixed run_id in one file without a plan.
- missing decision/boundary fields.
- transition happened but no world_transition event.
