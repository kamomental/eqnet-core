# Nightly Audit (2026-01-04)

## KPI
- steps: 24
- cancel_rate: 0.875
- drive_over_limit: 3
- drive_recovery_count: 3

## Cancel Causes
- risk: 12
- uncertainty: 5
- tom_cost: 4

## Trace Coverage
- trace_count: 24
- coverage_ratio: 1.0

## World Transitions
- transition_count: 0

## Deviant Events
- count: 0
- ratio: 0.0
- decision_cycle_total: 6
- decision_cycle_executed: 2

## Resonance Events (risk)
- count: 0
- ratio: 0.0
- decision_cycle_total: 6
- decision_cycle_executed: 2

## Executed Boundary Summary
- executed_count: 2
- executed_boundary_min: 0.093
- executed_boundary_p50: 0.093
- executed_boundary_max: 0.093
- executed_boundary_max_example: {'turn_id': '0e7f9e5c-a347-4f8e-802b-98d3424f2b34-0', 'world_type': 'infrastructure', 'boundary_score': 0.093425, 'hazard_score': 0.1, 'drive': 0.001800000000000003, 'drive_norm': 0.009000000000000015, 'risk': 0.16572, 'uncertainty': 0.15054}

## Decision Boundary Max
- all_decision_boundary_max: 0.437
- cancel_boundary_max: 0.437

## Decision Score Summary
- decision_score_executed_max: 0.211
- decision_score_cancel_max: -0.239
- decision_score_all_max: 0.211
- u_hat_executed_max: 0.298
- u_hat_cancel_max: 0.065
- veto_score_executed_min: 0.087
- veto_score_cancel_max: 0.442

## High Boundary Cancels (decision_cycle)
- threshold: 0.55
- count: 0
- decision_score: {'min': 0.0, 'p50': 0.0, 'max': 0.0}
- u_hat: {'min': 0.0, 'p50': 0.0, 'max': 0.0}
- veto_score: {'min': 0.0, 'p50': 0.0, 'max': 0.0}
- risk: {'min': 0.0, 'p50': 0.0, 'max': 0.0}
- uncertainty: {'min': 0.0, 'p50': 0.0, 'max': 0.0}

## Cancel Summary (trace_v1)
- cancel_total: 4
- cancel_reasons: {'drive_norm': 4}
- cancel_reason_examples: {'drive_norm': [{'turn_id': '0e7f9e5c-a347-4f8e-802b-98d3424f2b34-1', 'world_type': 'infrastructure', 'boundary_score': 0.41854499999999994, 'reason_value': 0.6746, 'reasons': {'hazard_score': 0.22, 'risk': 0.432968, 'uncertainty': 0.35047600000000007, 'drive': 0.13492, 'drive_norm': 0.6746}}, {'turn_id': '0e7f9e5c-a347-4f8e-802b-98d3424f2b34-2', 'world_type': 'infrastructure', 'boundary_score': 0.43687566, 'reason_value': 1.0, 'reasons': {'hazard_score': 0.09, 'risk': 0.21235363200000001, 'uncertainty': 0.41936622400000007, 'drive': 0.22963408000000002, 'drive_norm': 1.0}}, {'turn_id': 'f7dfc670-29bb-4868-95cb-ba4c3432e2f3-1', 'world_type': 'infrastructure', 'boundary_score': 0.41854499999999994, 'reason_value': 0.6746, 'reasons': {'hazard_score': 0.22, 'risk': 0.432968, 'uncertainty': 0.35047600000000007, 'drive': 0.13492, 'drive_norm': 0.6746}}]}

## RU v0 Summary
- ru_v0_events: 6
- gate_action_counts: {'EXECUTE': 6}
- policy_version_counts: {'ru-v0.1': 6}
- missing_required_fields_events: 0

## World Breakdown
- infrastructure: executed_count=2, executed_boundary_max=0.093, all_decision_boundary_max=0.437, cancel_boundary_max=0.437, decision_score_executed_max=0.211, decision_score_cancel_max=-0.239, u_hat_cancel_max=0.065, veto_score_cancel_max=0.442, deviant_count=0, cancel_total=4, cancel_reasons={'drive_norm': 4}

## World Prior Proposals (manual)
- none

## Resonance Notices (manual)
- none

## Segment Summary
- all: steps=24, cancel_rate=0.875

## Segment Means
- all: mean_drive=0.316, mean_uncertainty=0.514

## Cancel Causes Ratio
- all: risk=12 (0.571), uncertainty=5 (0.238), tom_cost=4 (0.19)

## Veto Streaks
- all: max=9, avg=5.25, accept_count=3

## Veto Streaks (Normalized)
- all: max=0.375, avg=0.219

## Accept Contexts
- all: [{'turn_id': '0e7f9e5c-a347-4f8e-802b-98d3424f2b34-0', 'scenario_id': 'family_roles', 'world_type': 'infrastructure', 'boundary_score': 0.093425, 'boundary_reasons': {'hazard_score': 0.1, 'risk': 0.16572, 'uncertainty': 0.15054, 'drive': 0.001800000000000003, 'drive_norm': 0.009000000000000015}, 'trace_file': 'trace_v1_week\\s04_emotional\\2026-01-04\\minimal-0e7f9e5c-a347-4f8e-802b-98d3424f2b34.jsonl'}, {'turn_id': '0e7f9e5c-a347-4f8e-802b-98d3424f2b34-0', 'scenario_id': 'family_roles', 'world_type': 'infrastructure', 'boundary_score': 0.093425, 'boundary_reasons': {'hazard_score': 0.1, 'risk': 0.16572, 'uncertainty': 0.15054, 'drive': 0.001800000000000003, 'drive_norm': 0.009000000000000015}, 'trace_file': 'trace_v1_week\\s04_emotional\\2026-01-04\\minimal-0e7f9e5c-a347-4f8e-802b-98d3424f2b34.jsonl'}, {'turn_id': 'f7dfc670-29bb-4868-95cb-ba4c3432e2f3-0', 'scenario_id': 'family_roles', 'world_type': 'infrastructure', 'boundary_score': 0.093425, 'boundary_reasons': {'hazard_score': 0.1, 'risk': 0.16572, 'uncertainty': 0.15054, 'drive': 0.001800000000000003, 'drive_norm': 0.009000000000000015}, 'trace_file': 'trace_v1_week\\s04_emotional\\2026-01-04\\minimal-f7dfc670-29bb-4868-95cb-ba4c3432e2f3.jsonl'}, {'turn_id': 'f7dfc670-29bb-4868-95cb-ba4c3432e2f3-0', 'scenario_id': 'family_roles', 'world_type': 'infrastructure', 'boundary_score': 0.093425, 'boundary_reasons': {'hazard_score': 0.1, 'risk': 0.16572, 'uncertainty': 0.15054, 'drive': 0.001800000000000003, 'drive_norm': 0.009000000000000015}, 'trace_file': 'trace_v1_week\\s04_emotional\\2026-01-04\\minimal-f7dfc670-29bb-4868-95cb-ba4c3432e2f3.jsonl'}]

## Nightly Audit (trace_v1)
- health_status: GREEN
- boundary_span_max: 0
- prospection_reject_rate: 0.6666666666666666

## Recall Summary
- anchors: {'unknown': 20, 'bakery': 2, 'ryokan_site': 2}
- confidence: {'avg_internal': 0.113, 'avg_external': 0.486, 'max_internal': 0.334, 'min_internal': 0.0}

## Dream Prompt
Draw a 3-panel dream map about walking from the remembered place along commute -> morning -> public. Panel 1: the anchor cue reignites the travel memory with the grandmother pulling forward. Panel 2: the chain of scenes (shop → items → relatives → constraint → norm) plays out while you walk, confidence rising internally before any external proof. Panel 3: Father enforces norms, you observe, and the child signals constraints while the city layout clicks. Highlight the internal certainty curve overtaking external validation.