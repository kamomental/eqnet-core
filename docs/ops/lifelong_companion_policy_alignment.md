# Lifelong Companion Policy Alignment (Mutualism)

## Purpose
- Confirm that `lifelong_companion_policy_v0` is consistent with existing EQNet docs and contracts.
- Fix the stance as mutualism: equal partner, no unilateral self-sacrifice, no isolation.

## Alignment Summary
- Aligned with approval-gated operations:
  - `docs/epics/epic6_policy_operations.md`
  - Policy changes are traceable and approval-based.
- Aligned with perception and privacy constraints:
  - `docs/perception_contract.md`
  - Plaintext minimization and source-aware handling remain mandatory.
- Aligned with imagery separation and anti-contamination:
  - `docs/ops/imagery_memory_separation_mvp.md`
  - Imagery remains isolated from factual/policy updates unless explicitly promoted.
- Aligned with arousal/safety downshift philosophy:
  - `docs/nightly_consciousness_and_arousal.md`
  - Safety and non-action paths are first-class behaviors.

## Mutualism Constraints Added
- Equal dignity and non-ownership are required.
- Human final decision and approval gate remain required.
- Non-isolation and reality-anchor are required.
- Self-sacrifice risk is treated as a blocker reason code.

## Contract Files
- `configs/lifelong_companion_policy_v0.yaml`
- `eqnet/runtime/companion_policy.py`

## Validation
- Contract test:
  - `tests/test_companion_policy_contract.py`
