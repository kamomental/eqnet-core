# ACE-aligned Weightless Operation for eqnet-core

## Purpose

This document defines how to improve `eqnet-core` without model weight updates.
It extracts ACE (Agentic Context Engineering) into practical ops rules that fit the
existing `trace_v1 -> nightly audit -> runtime overlay` loop.

## Non-goals

- No fine-tuning workflow is defined here.
- No long free-text lessons are stored as source of truth.
- No full prompt rewrite on every cycle.

## Core Principles (ACE)

1. Improve operating context, not model weights.
2. Apply local deltas, not full-context rewrites.
3. Encode lessons as contract keys, not free text.
4. Inject only relevant deltas at runtime.
5. Detect drift and conflicts via nightly audit.

## 4-layer Mapping in eqnet-core

### 1) Trace Store

- Existing: `trace_v1` for decisions, boundary, throttles, and acceptance.
- Extension: add `event_type=tool_call|micro_outcome`.
- Required fields (example):
  - `reason_codes`
  - `fingerprint`
  - `success`
  - `cost`

### 2) Reflector

- Existing: `scripts/run_nightly_audit.py` for aggregation and health checks.
- Recommended additions:
  - `micro_outcome_coverage`
  - `tool_failure_modes`
  - `delta_conflict_count`

### 3) Delta Curator

- Existing contract culture: `policy_version/profile/fingerprint/reason_codes`.
- Add append-only `rule_delta.v0` (`add|modify|disable`).
- Conflict resolution:
  - fingerprint conflict
  - priority
  - apply condition (`scenario/world_type/gate_action`)

### 4) Runtime Composer

- Existing: overlay-driven runtime application.
- Policy:
  - do not inject full rule sets by default
  - inject only condition-matched deltas
  - return injection result to trace for next audit cycle

## Weightless Daily Loop

1. Collect `trace_v1` during daytime runs.
2. Extract success/failure patterns in nightly audit.
3. Append or disable `rule_delta` entries (append-only).
4. Apply only condition-matched deltas via runtime overlay.
5. Escalate to `YELLOW/RED` when drift/conflict is detected.

## Security and Governance

- Prefer `reason_codes` over plain-language reasons.
- Do not write secrets or raw sensitive values into trace/delta.
- Treat missing contract keys as audit failures.
- Prefer `shadow -> canary -> on` rollout over direct full rollout.

## Minimal Adoption Checklist

- [ ] `trace_v1` records `tool_call|micro_outcome`.
- [ ] nightly audit outputs delta quality metrics.
- [ ] `rule_delta.v0` is managed as append-only.
- [ ] runtime selects deltas by explicit conditions.
- [ ] fail-closed behavior is covered by tests.

