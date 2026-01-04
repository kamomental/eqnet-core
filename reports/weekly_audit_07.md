# Nightly Audit (2026-01-04)

## KPI
- steps: 33
- cancel_rate: 0.848
- drive_over_limit: 3
- drive_recovery_count: 3

## Cancel Causes
- risk: 16
- uncertainty: 8
- tom_cost: 4

## Trace Coverage
- trace_count: 33
- coverage_ratio: 1.0

## World Transitions
- transition_count: 0

## Deviant Events
- count: 0
- ratio: 0.0
- decision_cycle_total: 4
- decision_cycle_executed: 0

## Resonance Events (risk)
- count: 0
- ratio: 0.0
- decision_cycle_total: 4
- decision_cycle_executed: 0

## Executed Boundary Summary
- executed_count: 0

## Decision Boundary Max
- all_decision_boundary_max: 0.397
- cancel_boundary_max: 0.397

## Decision Score Summary
- decision_score_executed_max: 0.0
- decision_score_cancel_max: -0.275
- decision_score_all_max: -0.275
- u_hat_executed_max: 0.0
- u_hat_cancel_max: 0.038
- veto_score_executed_min: 0.0
- veto_score_cancel_max: 0.656

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
- cancel_reasons: {'risk': 2, 'uncertainty': 2}
- cancel_reason_examples: {'risk': [{'turn_id': 'afebe00b-a760-4ef9-9686-6b7080470c88-0', 'world_type': 'infrastructure', 'boundary_score': 0.29551, 'reason_value': 0.58696, 'reasons': {'hazard_score': 0.25, 'risk': 0.58696, 'uncertainty': 0.35772000000000004, 'drive': 0.09240000000000001, 'drive_norm': 0.12320000000000002}}, {'turn_id': 'afebe00b-a760-4ef9-9686-6b7080470c88-2', 'world_type': 'infrastructure', 'boundary_score': 0.38466036584960006, 'reason_value': 0.6600549009749335, 'reasons': {'hazard_score': 0.2, 'risk': 0.6600549009749335, 'uncertainty': 0.5572965645312001, 'drive': 0.2476372524373334, 'drive_norm': 0.33018300324977784}}], 'uncertainty': [{'turn_id': 'afebe00b-a760-4ef9-9686-6b7080470c88-1', 'world_type': 'infrastructure', 'boundary_score': 0.23363046399999998, 'reason_value': 0.4390958080000001, 'reasons': {'hazard_score': 0.12, 'risk': 0.32480507733333336, 'uncertainty': 0.4390958080000001, 'drive': 0.15201269333333337, 'drive_norm': 0.20268359111111114}}, {'turn_id': 'afebe00b-a760-4ef9-9686-6b7080470c88-3', 'world_type': 'infrastructure', 'boundary_score': 0.3968191804526695, 'reason_value': 0.7059777908729039, 'reasons': {'hazard_score': 0.18, 'risk': 0.5199354265181116, 'uncertainty': 0.7059777908729039, 'drive': 0.309838566295279, 'drive_norm': 0.41311808839370534}}]}

## RU v0 Summary
- ru_v0_events: 4
- gate_action_counts: {'EXECUTE': 4}
- policy_version_counts: {'ru-v0.1': 4}
- missing_required_fields_events: 0

## World Breakdown
- infrastructure: executed_count=0, executed_boundary_max=0.0, all_decision_boundary_max=0.397, cancel_boundary_max=0.397, decision_score_executed_max=0.0, decision_score_cancel_max=-0.275, u_hat_cancel_max=0.038, veto_score_cancel_max=0.656, deviant_count=0, cancel_total=4, cancel_reasons={'risk': 2, 'uncertainty': 2}

## World Prior Proposals (manual)
- none

## Resonance Notices (manual)
- none

## Segment Summary
- all: steps=33, cancel_rate=0.848

## Segment Means
- all: mean_drive=0.274, mean_uncertainty=0.499

## Cancel Causes Ratio
- all: risk=16 (0.571), uncertainty=8 (0.286), tom_cost=4 (0.143)

## Veto Streaks
- all: max=9, avg=5.6, accept_count=5

## Veto Streaks (Normalized)
- all: max=0.273, avg=0.17

## Accept Contexts

## Nightly Audit (trace_v1)
- health_status: YELLOW
- boundary_span_max: 0
- prospection_reject_rate: 1.0

## Recall Summary
- anchors: {'unknown': 26, 'bakery': 4, 'ryokan_site': 3}
- confidence: {'avg_internal': 0.108, 'avg_external': 0.501, 'max_internal': 0.386, 'min_internal': 0.0}

## Dream Prompt
Draw a 3-panel dream map about walking from the remembered place along commute -> morning -> public. Panel 1: the anchor cue reignites the travel memory with the grandmother pulling forward. Panel 2: the chain of scenes (shop → items → relatives → constraint → norm) plays out while you walk, confidence rising internally before any external proof. Panel 3: Father enforces norms, you observe, and the child signals constraints while the city layout clicks. Highlight the internal certainty curve overtaking external validation.