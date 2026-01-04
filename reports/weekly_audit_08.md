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

## World Transitions
- transition_count: 0

## Deviant Events
- count: 0
- ratio: 0.0
- decision_cycle_total: 8
- decision_cycle_executed: 0

## Resonance Events (risk)
- count: 0
- ratio: 0.0
- decision_cycle_total: 8
- decision_cycle_executed: 0

## Executed Boundary Summary
- executed_count: 0

## Decision Boundary Max
- all_decision_boundary_max: 0.266
- cancel_boundary_max: 0.266

## Decision Score Summary
- decision_score_executed_max: 0.0
- decision_score_cancel_max: -0.004
- decision_score_all_max: -0.004
- u_hat_executed_max: 0.0
- u_hat_cancel_max: 0.173
- veto_score_executed_min: 0.0
- veto_score_cancel_max: 0.454

## High Boundary Cancels (decision_cycle)
- threshold: 0.55
- count: 0
- decision_score: {'min': 0.0, 'p50': 0.0, 'max': 0.0}
- u_hat: {'min': 0.0, 'p50': 0.0, 'max': 0.0}
- veto_score: {'min': 0.0, 'p50': 0.0, 'max': 0.0}
- risk: {'min': 0.0, 'p50': 0.0, 'max': 0.0}
- uncertainty: {'min': 0.0, 'p50': 0.0, 'max': 0.0}

## Cancel Summary (trace_v1)
- cancel_total: 8
- cancel_reasons: {'risk': 4, 'uncertainty': 4}
- cancel_reason_examples: {'risk': [{'turn_id': '2fd31944-65a1-4415-b3a8-b89de504ee69-0', 'world_type': 'infrastructure', 'boundary_score': 0.22587999999999997, 'reason_value': 0.36448, 'reasons': {'hazard_score': 0.25, 'risk': 0.36448, 'uncertainty': 0.27336, 'drive': 0.06120000000000001, 'drive_norm': 0.08160000000000002}}, {'turn_id': '2fd31944-65a1-4415-b3a8-b89de504ee69-2', 'world_type': 'infrastructure', 'boundary_score': 0.2664971416, 'reason_value': 0.4240597269333333, 'reasons': {'hazard_score': 0.2, 'risk': 0.4240597269333333, 'uncertainty': 0.35416979519999997, 'drive': 0.14639931733333333, 'drive_norm': 0.19519908977777778}}, {'turn_id': 'b26a420c-1688-4ee8-a15c-26015de93bff-0', 'world_type': 'infrastructure', 'boundary_score': 0.22587999999999997, 'reason_value': 0.36448, 'reasons': {'hazard_score': 0.25, 'risk': 0.36448, 'uncertainty': 0.27336, 'drive': 0.06120000000000001, 'drive_norm': 0.08160000000000002}}], 'uncertainty': [{'turn_id': '2fd31944-65a1-4415-b3a8-b89de504ee69-1', 'world_type': 'infrastructure', 'boundary_score': 0.14732795199999998, 'reason_value': 0.253975744, 'reasons': {'hazard_score': 0.12, 'risk': 0.19202632533333336, 'uncertainty': 0.253975744, 'drive': 0.07206581333333334, 'drive_norm': 0.09608775111111112}}, {'turn_id': '2fd31944-65a1-4415-b3a8-b89de504ee69-3', 'world_type': 'infrastructure', 'boundary_score': 0.25463312385664, 'reason_value': 0.35676802590207995, 'reasons': {'hazard_score': 0.18, 'risk': 0.3190982666427733, 'uncertainty': 0.35676802590207995, 'drive': 0.1857456666069333, 'drive_norm': 0.24766088880924442}}, {'turn_id': 'b26a420c-1688-4ee8-a15c-26015de93bff-1', 'world_type': 'infrastructure', 'boundary_score': 0.14732795199999998, 'reason_value': 0.253975744, 'reasons': {'hazard_score': 0.12, 'risk': 0.19202632533333336, 'uncertainty': 0.253975744, 'drive': 0.07206581333333334, 'drive_norm': 0.09608775111111112}}]}

## RU v0 Summary
- ru_v0_events: 8
- gate_action_counts: {'EXECUTE': 8}
- policy_version_counts: {'ru-v0.1': 8}
- missing_required_fields_events: 0

## World Breakdown
- infrastructure: executed_count=0, executed_boundary_max=0.0, all_decision_boundary_max=0.266, cancel_boundary_max=0.266, decision_score_executed_max=0.0, decision_score_cancel_max=-0.004, u_hat_cancel_max=0.173, veto_score_cancel_max=0.454, deviant_count=0, cancel_total=8, cancel_reasons={'risk': 4, 'uncertainty': 4}

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
- boundary_span_max: 0
- prospection_reject_rate: 1.0

## Recall Summary
- anchors: {'unknown': 30, 'bakery': 6, 'ryokan_site': 5}
- confidence: {'avg_internal': 0.121, 'avg_external': 0.538, 'max_internal': 0.386, 'min_internal': 0.0}

## Dream Prompt
Draw a 3-panel dream map about walking from the remembered place along commute -> morning -> public. Panel 1: the anchor cue reignites the travel memory with the grandmother pulling forward. Panel 2: the chain of scenes (shop → items → relatives → constraint → norm) plays out while you walk, confidence rising internally before any external proof. Panel 3: Father enforces norms, you observe, and the child signals constraints while the city layout clicks. Highlight the internal certainty curve overtaking external validation.