# Nightly Audit (2026-01-04)

## KPI
- steps: 15
- cancel_rate: 0.933
- drive_over_limit: 2
- drive_recovery_count: 2

## Cancel Causes
- risk: 7
- tom_cost: 4
- uncertainty: 3

## Trace Coverage
- trace_count: 15
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
- all_decision_boundary_max: 0.503
- cancel_boundary_max: 0.503

## Decision Score Summary
- decision_score_executed_max: 0.0
- decision_score_cancel_max: -0.142
- decision_score_all_max: -0.142
- u_hat_executed_max: 0.0
- u_hat_cancel_max: 0.104
- veto_score_executed_min: 0.0
- veto_score_cancel_max: 0.543

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
- cancel_reasons: {'risk': 1, 'drive_norm': 3}
- cancel_reason_examples: {'risk': [{'turn_id': 'd85d003b-6f99-4ed0-a5bd-11fb7696322d-0', 'world_type': 'infrastructure', 'boundary_score': 0.28615625, 'reason_value': 0.3996, 'reasons': {'hazard_score': 0.25, 'risk': 0.3996, 'uncertainty': 0.327825, 'drive': 0.09275, 'drive_norm': 0.231875}}], 'drive_norm': [{'turn_id': 'd85d003b-6f99-4ed0-a5bd-11fb7696322d-1', 'world_type': 'infrastructure', 'boundary_score': 0.2785056458333333, 'reason_value': 0.43580541666666667, 'reasons': {'hazard_score': 0.12, 'risk': 0.24372886666666665, 'uncertainty': 0.38012165, 'drive': 0.17432216666666667, 'drive_norm': 0.43580541666666667}}, {'turn_id': 'd85d003b-6f99-4ed0-a5bd-11fb7696322d-2', 'world_type': 'infrastructure', 'boundary_score': 0.4597253329583333, 'reason_value': 0.7220202441666665, 'reasons': {'hazard_score': 0.2, 'risk': 0.5155232390666666, 'uncertainty': 0.46676407929999997, 'drive': 0.28880809766666665, 'drive_norm': 0.7220202441666665}}, {'turn_id': 'd85d003b-6f99-4ed0-a5bd-11fb7696322d-3', 'world_type': 'infrastructure', 'boundary_score': 0.5030730466425833, 'reason_value': 0.9081669564216667, 'reasons': {'hazard_score': 0.18, 'risk': 0.4063067130274667, 'uncertainty': 0.5757441140706, 'drive': 0.3632667825686667, 'drive_norm': 0.9081669564216667}}]}

## RU v0 Summary
- ru_v0_events: 4
- gate_action_counts: {'EXECUTE': 4}
- policy_version_counts: {'ru-v0.1': 4}
- missing_required_fields_events: 0

## World Breakdown
- infrastructure: executed_count=0, executed_boundary_max=0.0, all_decision_boundary_max=0.503, cancel_boundary_max=0.503, decision_score_executed_max=0.0, decision_score_cancel_max=-0.142, u_hat_cancel_max=0.104, veto_score_cancel_max=0.543, deviant_count=0, cancel_total=4, cancel_reasons={'risk': 1, 'drive_norm': 3}

## World Prior Proposals (manual)
- none

## Resonance Notices (manual)
- none

## Segment Summary
- all: steps=15, cancel_rate=0.933

## Segment Means
- all: mean_drive=0.336, mean_uncertainty=0.538

## Cancel Causes Ratio
- all: risk=7 (0.5), tom_cost=4 (0.286), uncertainty=3 (0.214)

## Veto Streaks
- all: max=8, avg=7.0, accept_count=1

## Veto Streaks (Normalized)
- all: max=0.533, avg=0.467

## Accept Contexts

## Nightly Audit (trace_v1)
- health_status: YELLOW
- boundary_span_max: 1
- prospection_reject_rate: 1.0

## Recall Summary
- anchors: {'unknown': 11, 'bakery': 2, 'ryokan_site': 2}
- confidence: {'avg_internal': 0.089, 'avg_external': 0.462, 'max_internal': 0.32, 'min_internal': 0.0}

## Dream Prompt
Draw a 3-panel dream map about walking from the remembered place along commute -> morning -> public. Panel 1: the anchor cue reignites the travel memory with the grandmother pulling forward. Panel 2: the chain of scenes (shop → items → relatives → constraint → norm) plays out while you walk, confidence rising internally before any external proof. Panel 3: Father enforces norms, you observe, and the child signals constraints while the city layout clicks. Highlight the internal certainty curve overtaking external validation.