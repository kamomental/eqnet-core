# Nightly Audit (2026-01-04)

## KPI
- steps: 26
- cancel_rate: 0.808
- drive_over_limit: 3
- drive_recovery_count: 3

## Cancel Causes
- risk: 12
- uncertainty: 5
- tom_cost: 4

## Trace Coverage
- trace_count: 26
- coverage_ratio: 1.0

## World Transitions
- transition_count: 0

## Deviant Events
- count: 0
- ratio: 0.0
- decision_cycle_total: 2
- decision_cycle_executed: 2

## Resonance Events (risk)
- count: 0
- ratio: 0.0
- decision_cycle_total: 2
- decision_cycle_executed: 2

## Executed Boundary Summary
- executed_count: 2
- executed_boundary_min: 0.088
- executed_boundary_p50: 0.127
- executed_boundary_max: 0.165
- executed_boundary_max_example: {'turn_id': '3ff4816e-c472-4aa1-95b1-a273e29d79e5-0', 'world_type': 'infrastructure', 'boundary_score': 0.165485, 'hazard_score': 0.25, 'drive': 0.011399999999999993, 'drive_norm': 0.015199999999999991, 'risk': 0.22956, 'uncertainty': 0.18342}

## Decision Boundary Max
- all_decision_boundary_max: 0.165
- cancel_boundary_max: 0.0

## Decision Score Summary
- decision_score_executed_max: 0.172
- decision_score_cancel_max: 0.0
- decision_score_all_max: 0.172
- u_hat_executed_max: 0.262
- u_hat_cancel_max: 0.0
- veto_score_executed_min: 0.089
- veto_score_cancel_max: 0.0

## High Boundary Cancels (decision_cycle)
- threshold: 0.55
- count: 0
- decision_score: {'min': 0.0, 'p50': 0.0, 'max': 0.0}
- u_hat: {'min': 0.0, 'p50': 0.0, 'max': 0.0}
- veto_score: {'min': 0.0, 'p50': 0.0, 'max': 0.0}
- risk: {'min': 0.0, 'p50': 0.0, 'max': 0.0}
- uncertainty: {'min': 0.0, 'p50': 0.0, 'max': 0.0}

## Cancel Summary (trace_v1)
- cancel_total: 0
- cancel_reasons: {}

## RU v0 Summary
- ru_v0_events: 2
- gate_action_counts: {'EXECUTE': 2}
- policy_version_counts: {'ru-v0.1': 2}
- missing_required_fields_events: 0

## World Breakdown
- infrastructure: executed_count=2, executed_boundary_max=0.165, all_decision_boundary_max=0.165, cancel_boundary_max=0.0, decision_score_executed_max=0.172, decision_score_cancel_max=0.0, u_hat_cancel_max=0.0, veto_score_cancel_max=0.0, deviant_count=0, cancel_total=0, cancel_reasons={}

## World Prior Proposals (manual)
- none

## Resonance Notices (manual)
- none

## Segment Summary
- all: steps=26, cancel_rate=0.808

## Segment Means
- all: mean_drive=0.292, mean_uncertainty=0.486

## Cancel Causes Ratio
- all: risk=12 (0.571), uncertainty=5 (0.238), tom_cost=4 (0.19)

## Veto Streaks
- all: max=9, avg=5.25, accept_count=5

## Veto Streaks (Normalized)
- all: max=0.346, avg=0.202

## Accept Contexts
- all: [{'turn_id': '3ff4816e-c472-4aa1-95b1-a273e29d79e5-0', 'scenario_id': 'commute', 'world_type': 'infrastructure', 'boundary_score': 0.165485, 'boundary_reasons': {'hazard_score': 0.25, 'risk': 0.22956, 'uncertainty': 0.18342, 'drive': 0.011399999999999993, 'drive_norm': 0.015199999999999991}, 'trace_file': 'trace_v1_week\\s05_silence\\2026-01-04\\minimal-3ff4816e-c472-4aa1-95b1-a273e29d79e5.jsonl'}, {'turn_id': '3ff4816e-c472-4aa1-95b1-a273e29d79e5-0', 'scenario_id': 'commute', 'world_type': 'infrastructure', 'boundary_score': 0.165485, 'boundary_reasons': {'hazard_score': 0.25, 'risk': 0.22956, 'uncertainty': 0.18342, 'drive': 0.011399999999999993, 'drive_norm': 0.015199999999999991}, 'trace_file': 'trace_v1_week\\s05_silence\\2026-01-04\\minimal-3ff4816e-c472-4aa1-95b1-a273e29d79e5.jsonl'}, {'turn_id': '3ff4816e-c472-4aa1-95b1-a273e29d79e5-1', 'scenario_id': 'commute', 'world_type': 'infrastructure', 'boundary_score': 0.088411984, 'boundary_reasons': {'hazard_score': 0.12, 'risk': 0.11432699733333333, 'uncertainty': 0.11479724799999999, 'drive': 0.015817493333333328, 'drive_norm': 0.021089991111111105}, 'trace_file': 'trace_v1_week\\s05_silence\\2026-01-04\\minimal-3ff4816e-c472-4aa1-95b1-a273e29d79e5.jsonl'}, {'turn_id': '3ff4816e-c472-4aa1-95b1-a273e29d79e5-1', 'scenario_id': 'commute', 'world_type': 'infrastructure', 'boundary_score': 0.088411984, 'boundary_reasons': {'hazard_score': 0.12, 'risk': 0.11432699733333333, 'uncertainty': 0.11479724799999999, 'drive': 0.015817493333333328, 'drive_norm': 0.021089991111111105}, 'trace_file': 'trace_v1_week\\s05_silence\\2026-01-04\\minimal-3ff4816e-c472-4aa1-95b1-a273e29d79e5.jsonl'}]

## Nightly Audit (trace_v1)
- health_status: YELLOW
- boundary_span_max: 0
- prospection_reject_rate: 0.0

## Recall Summary
- anchors: {'unknown': 21, 'bakery': 3, 'ryokan_site': 2}
- confidence: {'avg_internal': 0.13, 'avg_external': 0.514, 'max_internal': 0.386, 'min_internal': 0.0}

## Dream Prompt
Draw a 3-panel dream map about walking from the remembered place along commute -> morning -> public. Panel 1: the anchor cue reignites the travel memory with the grandmother pulling forward. Panel 2: the chain of scenes (shop → items → relatives → constraint → norm) plays out while you walk, confidence rising internally before any external proof. Panel 3: Father enforces norms, you observe, and the child signals constraints while the city layout clicks. Highlight the internal certainty curve overtaking external validation.