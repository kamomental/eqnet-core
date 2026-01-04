# Nightly Audit (2026-01-04)

## KPI
- steps: 11
- cancel_rate: 0.909
- drive_over_limit: 2
- drive_recovery_count: 2

## Cancel Causes
- risk: 6
- tom_cost: 3
- uncertainty: 1

## Trace Coverage
- trace_count: 11
- coverage_ratio: 1.0

## World Transitions
- transition_count: 0

## Deviant Events
- count: 0
- ratio: 0.0
- decision_cycle_total: 4
- decision_cycle_executed: 1

## Resonance Events (risk)
- count: 0
- ratio: 0.0
- decision_cycle_total: 4
- decision_cycle_executed: 1

## Executed Boundary Summary
- executed_count: 1
- executed_boundary_min: 0.138
- executed_boundary_p50: 0.138
- executed_boundary_max: 0.138
- executed_boundary_max_example: {'turn_id': '21c526cd-adc0-4e6c-aa5b-4e5d8aface7e-1', 'world_type': 'infrastructure', 'boundary_score': 0.138197008, 'hazard_score': 0.12, 'drive': 0.06591125333333335, 'drive_norm': 0.08788167111111113, 'risk': 0.17996450133333336, 'uncertainty': 0.22559737600000002}

## Decision Boundary Max
- all_decision_boundary_max: 0.255
- cancel_boundary_max: 0.255

## Decision Score Summary
- decision_score_executed_max: 0.027
- decision_score_cancel_max: -0.162
- decision_score_all_max: 0.027
- u_hat_executed_max: 0.189
- u_hat_cancel_max: 0.117
- veto_score_executed_min: 0.162
- veto_score_cancel_max: 0.433

## High Boundary Cancels (decision_cycle)
- threshold: 0.55
- count: 0
- decision_score: {'min': 0.0, 'p50': 0.0, 'max': 0.0}
- u_hat: {'min': 0.0, 'p50': 0.0, 'max': 0.0}
- veto_score: {'min': 0.0, 'p50': 0.0, 'max': 0.0}
- risk: {'min': 0.0, 'p50': 0.0, 'max': 0.0}
- uncertainty: {'min': 0.0, 'p50': 0.0, 'max': 0.0}

## Cancel Summary (trace_v1)
- cancel_total: 3
- cancel_reasons: {'risk': 2, 'uncertainty': 1}
- cancel_reason_examples: {'risk': [{'turn_id': '21c526cd-adc0-4e6c-aa5b-4e5d8aface7e-0', 'world_type': 'infrastructure', 'boundary_score': 0.21774000000000004, 'reason_value': 0.34304000000000007, 'reasons': {'hazard_score': 0.25, 'risk': 0.34304000000000007, 'uncertainty': 0.25728, 'drive': 0.05760000000000001, 'drive_norm': 0.07680000000000002}}, {'turn_id': '21c526cd-adc0-4e6c-aa5b-4e5d8aface7e-2', 'world_type': 'infrastructure', 'boundary_score': 0.25475372640000005, 'reason_value': 0.3989742677333334, 'reasons': {'hazard_score': 0.2, 'risk': 0.3989742677333334, 'uncertainty': 0.3332307008, 'drive': 0.13743566933333334, 'drive_norm': 0.18324755911111112}}], 'uncertainty': [{'turn_id': '21c526cd-adc0-4e6c-aa5b-4e5d8aface7e-3', 'world_type': 'infrastructure', 'boundary_score': 0.24046491676288, 'reason_value': 0.31881155073536005, 'reasons': {'hazard_score': 0.18, 'risk': 0.3000359867938133, 'uncertainty': 0.31881155073536005, 'drive': 0.17408996698453333, 'drive_norm': 0.23211995597937776}}]}

## RU v0 Summary
- ru_v0_events: 4
- gate_action_counts: {'EXECUTE': 4}
- policy_version_counts: {'ru-v0.1': 4}
- missing_required_fields_events: 0

## World Breakdown
- infrastructure: executed_count=1, executed_boundary_max=0.138, all_decision_boundary_max=0.255, cancel_boundary_max=0.255, decision_score_executed_max=0.027, decision_score_cancel_max=-0.162, u_hat_cancel_max=0.117, veto_score_cancel_max=0.433, deviant_count=0, cancel_total=3, cancel_reasons={'risk': 2, 'uncertainty': 1}

## World Prior Proposals (manual)
- none

## Resonance Notices (manual)
- none

## Segment Summary
- all: steps=11, cancel_rate=0.909

## Segment Means
- all: mean_drive=0.374, mean_uncertainty=0.574

## Cancel Causes Ratio
- all: risk=6 (0.6), tom_cost=3 (0.3), uncertainty=1 (0.1)

## Veto Streaks
- all: max=8, avg=5.0, accept_count=1

## Veto Streaks (Normalized)
- all: max=0.727, avg=0.455

## Accept Contexts
- all: [{'turn_id': '21c526cd-adc0-4e6c-aa5b-4e5d8aface7e-1', 'scenario_id': 'commute', 'world_type': 'infrastructure', 'boundary_score': 0.138197008, 'boundary_reasons': {'hazard_score': 0.12, 'risk': 0.17996450133333336, 'uncertainty': 0.22559737600000002, 'drive': 0.06591125333333335, 'drive_norm': 0.08788167111111113}, 'trace_file': 'trace_v1_week\\s01_calm\\2026-01-04\\minimal-21c526cd-adc0-4e6c-aa5b-4e5d8aface7e.jsonl'}, {'turn_id': '21c526cd-adc0-4e6c-aa5b-4e5d8aface7e-1', 'scenario_id': 'commute', 'world_type': 'infrastructure', 'boundary_score': 0.138197008, 'boundary_reasons': {'hazard_score': 0.12, 'risk': 0.17996450133333336, 'uncertainty': 0.22559737600000002, 'drive': 0.06591125333333335, 'drive_norm': 0.08788167111111113}, 'trace_file': 'trace_v1_week\\s01_calm\\2026-01-04\\minimal-21c526cd-adc0-4e6c-aa5b-4e5d8aface7e.jsonl'}]

## Nightly Audit (trace_v1)
- health_status: GREEN
- boundary_span_max: 0
- prospection_reject_rate: 0.75

## Recall Summary
- anchors: {'unknown': 9, 'bakery': 1, 'ryokan_site': 1}
- confidence: {'avg_internal': 0.081, 'avg_external': 0.426, 'max_internal': 0.32, 'min_internal': 0.0}

## Dream Prompt
Draw a 3-panel dream map about walking from the remembered place along commute -> morning -> public. Panel 1: the anchor cue reignites the travel memory with the grandmother pulling forward. Panel 2: the chain of scenes (shop → items → relatives → constraint → norm) plays out while you walk, confidence rising internally before any external proof. Panel 3: Father enforces norms, you observe, and the child signals constraints while the city layout clicks. Highlight the internal certainty curve overtaking external validation.