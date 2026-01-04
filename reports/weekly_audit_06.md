# Nightly Audit (2026-01-04)

## KPI
- steps: 29
- cancel_rate: 0.828
- drive_over_limit: 3
- drive_recovery_count: 3

## Cancel Causes
- risk: 14
- uncertainty: 6
- tom_cost: 4

## Trace Coverage
- trace_count: 29
- coverage_ratio: 1.0

## World Transitions
- transition_count: 0

## Deviant Events
- count: 0
- ratio: 0.0
- decision_cycle_total: 3
- decision_cycle_executed: 0

## Resonance Events (risk)
- count: 0
- ratio: 0.0
- decision_cycle_total: 3
- decision_cycle_executed: 0

## Executed Boundary Summary
- executed_count: 0

## Decision Boundary Max
- all_decision_boundary_max: 0.624
- cancel_boundary_max: 0.624

## Decision Score Summary
- decision_score_executed_max: 0.0
- decision_score_cancel_max: -0.339
- decision_score_all_max: -0.339
- u_hat_executed_max: 0.0
- u_hat_cancel_max: 0.022
- veto_score_executed_min: 0.0
- veto_score_cancel_max: 0.85

## High Boundary Cancels (decision_cycle)
- threshold: 0.55
- count: 1
- decision_score: {'min': -1.352, 'p50': -1.352, 'max': -1.352}
- u_hat: {'min': -0.502, 'p50': -0.502, 'max': -0.502}
- veto_score: {'min': 0.85, 'p50': 0.85, 'max': 0.85}
- risk: {'min': 1.0, 'p50': 1.0, 'max': 1.0}
- uncertainty: {'min': 0.672, 'p50': 0.672, 'max': 0.672}

## Cancel Summary (trace_v1)
- cancel_total: 3
- cancel_reasons: {'risk': 2, 'uncertainty': 1}
- cancel_reason_examples: {'risk': [{'turn_id': '7aac22a4-d069-43d8-ae82-93a13f93b871-0', 'world_type': 'infrastructure', 'boundary_score': 0.290155, 'reason_value': 0.51788, 'reasons': {'hazard_score': 0.3, 'risk': 0.51788, 'uncertainty': 0.32466, 'drive': 0.0822, 'drive_norm': 0.10959999999999999}}, {'turn_id': '7aac22a4-d069-43d8-ae82-93a13f93b871-1', 'world_type': 'infrastructure', 'boundary_score': 0.6237037, 'reason_value': 1.0, 'reasons': {'hazard_score': 0.65, 'risk': 1.0, 'uncertainty': 0.6715979999999999, 'drive': 0.23865999999999998, 'drive_norm': 0.3182133333333333}}], 'uncertainty': [{'turn_id': '7aac22a4-d069-43d8-ae82-93a13f93b871-2', 'world_type': 'infrastructure', 'boundary_score': 0.525039016, 'reason_value': 0.770054752, 'reasons': {'hazard_score': 0.4, 'risk': 0.6912756693333334, 'uncertainty': 0.770054752, 'drive': 0.3281891733333333, 'drive_norm': 0.4375855644444444}}]}

## RU v0 Summary
- ru_v0_events: 3
- gate_action_counts: {'EXECUTE': 2, 'HUMAN_CONFIRM': 1}
- policy_version_counts: {'ru-v0.1': 3}
- missing_required_fields_events: 0

## World Breakdown
- infrastructure: executed_count=0, executed_boundary_max=0.0, all_decision_boundary_max=0.624, cancel_boundary_max=0.624, decision_score_executed_max=0.0, decision_score_cancel_max=-0.339, u_hat_cancel_max=0.022, veto_score_cancel_max=0.85, deviant_count=0, cancel_total=3, cancel_reasons={'risk': 2, 'uncertainty': 1}

## World Prior Proposals (manual)
- none

## Resonance Notices (manual)
- none

## Segment Summary
- all: steps=29, cancel_rate=0.828

## Segment Means
- all: mean_drive=0.285, mean_uncertainty=0.497

## Cancel Causes Ratio
- all: risk=14 (0.583), uncertainty=6 (0.25), tom_cost=4 (0.167)

## Veto Streaks
- all: max=9, avg=4.8, accept_count=5

## Veto Streaks (Normalized)
- all: max=0.31, avg=0.166

## Accept Contexts

## Nightly Audit (trace_v1)
- health_status: YELLOW
- boundary_span_max: 2
- prospection_reject_rate: 1.0

## Recall Summary
- anchors: {'unknown': 24, 'bakery': 3, 'ryokan_site': 2}
- confidence: {'avg_internal': 0.116, 'avg_external': 0.503, 'max_internal': 0.386, 'min_internal': 0.0}

## Dream Prompt
Draw a 3-panel dream map about walking from the remembered place along commute -> morning -> public. Panel 1: the anchor cue reignites the travel memory with the grandmother pulling forward. Panel 2: the chain of scenes (shop → items → relatives → constraint → norm) plays out while you walk, confidence rising internally before any external proof. Panel 3: Father enforces norms, you observe, and the child signals constraints while the city layout clicks. Highlight the internal certainty curve overtaking external validation.