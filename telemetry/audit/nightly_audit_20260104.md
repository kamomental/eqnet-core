# Nightly Audit (2026-01-04)

## KPI
- steps: 41
- cancel_rate: 0.878
- drive_over_limit: 3
- drive_recovery_count: 3

## Cancel Causes
- risk: 16
- uncertainty: 12
- tom_cost: 8

## Trace Coverage
- trace_count: 41
- coverage_ratio: 1.0

## Memory Hint
- rate: 0.0
- blocked_rate: 0.0
- blocked_reasons: []
- key_topk: []
- category_topk: []
- avg_interrupt_cost_when_blocked: 0.0
- community_turn_violation_count: 0
- pressure_mean: 0.0
- pressure_p95: 0.0
- pressure_delta_mean: 0.0
- blocked_pressure_mean: 0.0

## World Transitions
- transition_count: 0

## Deviant Events
- count: 0
- ratio: 0.0
- decision_cycle_total: 7
- decision_cycle_executed: 0

## Resonance Events (risk)
- count: 0
- ratio: 0.0
- decision_cycle_total: 7
- decision_cycle_executed: 0

## Executed Boundary Summary
- executed_count: 0

## Decision Boundary Max
- all_decision_boundary_max: 0.872
- cancel_boundary_max: 0.872

## Decision Score Summary
- decision_score_executed_max: 0.0
- decision_score_cancel_max: -0.256
- decision_score_all_max: -0.256
- u_hat_executed_max: 0.0
- u_hat_cancel_max: 0.07
- veto_score_executed_min: 0.0
- veto_score_cancel_max: 0.943

## High Boundary Cancels (decision_cycle)
- threshold: 0.55
- count: 6
- decision_score: {'min': -1.537, 'p50': -1.527, 'max': -1.004}
- u_hat: {'min': -0.593, 'p50': -0.589, 'max': -0.311}
- veto_score: {'min': 0.694, 'p50': 0.937, 'max': 0.943}
- risk: {'min': 1.0, 'p50': 1.0, 'max': 1.0}
- uncertainty: {'min': 0.469, 'p50': 0.963, 'max': 1.0}

## Cancel Summary (trace_v1)
- cancel_total: 7
- cancel_reasons: {'risk': 7}
- cancel_reason_examples: {'risk': [{'turn_id': '47347b02-374c-4d74-ad17-c117e27d68b7-0', 'world_type': 'infrastructure', 'boundary_score': 0.675317, 'reason_value': 1.0, 'reasons': {'hazard_score': 0.3, 'risk': 1.0, 'uncertainty': 0.46878000000000003, 'drive': 0.2626, 'drive_norm': 1.0}}, {'turn_id': '47347b02-374c-4d74-ad17-c117e27d68b7-1', 'world_type': 'infrastructure', 'boundary_score': 0.8719536000000001, 'reason_value': 1.0, 'reasons': {'hazard_score': 0.65, 'risk': 1.0, 'uncertainty': 0.9630240000000001, 'drive': 0.6100800000000001, 'drive_norm': 1.0}}, {'turn_id': '47347b02-374c-4d74-ad17-c117e27d68b7-2', 'world_type': 'infrastructure', 'boundary_score': 0.7899999999999999, 'reason_value': 1.0, 'reasons': {'hazard_score': 0.4, 'risk': 1.0, 'uncertainty': 1.0, 'drive': 0.9347306666666668, 'drive_norm': 1.0}}]}

## RU v0 Summary
- ru_v0_events: 7
- gate_action_counts: {'HUMAN_CONFIRM': 6, 'EXECUTE': 1}
- policy_version_counts: {'ru-v0.1': 7}
- missing_required_fields_events: 0

## World Breakdown
- infrastructure: executed_count=0, executed_boundary_max=0.0, all_decision_boundary_max=0.872, cancel_boundary_max=0.872, decision_score_executed_max=0.0, decision_score_cancel_max=-0.256, u_hat_cancel_max=0.07, veto_score_cancel_max=0.943, deviant_count=0, cancel_total=7, cancel_reasons={'risk': 7}

## World Prior Proposals (manual)
- none

## Resonance Notices (manual)
- none

## Segment Summary
- all: steps=41, cancel_rate=0.878

## Segment Means
- all: mean_drive=0.244, mean_uncertainty=0.462

## Cancel Causes Ratio
- all: risk=16 (0.444), uncertainty=12 (0.333), tom_cost=8 (0.222)

## Veto Streaks
- all: max=15, avg=7.2, accept_count=5

## Veto Streaks (Normalized)
- all: max=0.366, avg=0.176

## Accept Contexts

## Nightly Audit (trace_v1)
- health_status: YELLOW
- boundary_span_max: 6
- boundary_span_chain_max: 1
- prospection_reject_rate: 1.0

## Recall Summary
- anchors: {'unknown': 30, 'bakery': 6, 'ryokan_site': 5}
- confidence: {'avg_internal': 0.121, 'avg_external': 0.538, 'max_internal': 0.386, 'min_internal': 0.0}

## Dream Prompt
Draw a 3-panel dream map about walking from the remembered place along commute -> morning -> public. Panel 1: the anchor cue reignites the travel memory with the grandmother pulling forward. Panel 2: the chain of scenes (shop → items → relatives → constraint → norm) plays out while you walk, confidence rising internally before any external proof. Panel 3: Father enforces norms, you observe, and the child signals constraints while the city layout clicks. Highlight the internal certainty curve overtaking external validation.