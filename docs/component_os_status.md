# Component OS Status

This document tracks the current state of the reusable component OS view.

## Complete

- A first sleep-like bridge now exists outside `inner_os`, wrapping legacy
  rest/field/nightly signals into the reusable sleep-consolidation snapshot
  boundary instead of forcing `inner_os` to import repo-specific lifecycle code.
- `inner_os/` exists as a root-level reusable receiver.
- Core modules exist for physiology, temporal weighting, conscious access,
  relational world, memory, terrain, simulation transfer, and working memory.
- Explicit hook contracts exist for `pre_turn_update`, `memory_recall`,
  `response_gate`, and `post_turn_update`.
- `InnerOSService` now exposes those hooks through a plain mapping-to-dict
  boundary.
- `EmotionalHubRuntime.process_turn()` already uses the hook path internally.
- Runtime-side memory references and observed-vision entries now bridge into
  `inner_os` memory classes through one OS-side helper path.
- Recalled memory can now be reinterpreted and reconsolidated inside
  `inner_os`, instead of leaving memory reframing implicit in runtime-only
  behavior.
- Environment pressure now feeds the same OS path, so scarcity, hazard,
  ritual, and institutional pressure can bias development and memory
  reinterpretation without relying on UI-only summaries.
- Repeated relationship signals now feed OS-side personality sediment, so
  attachment and familiarity can accumulate into longer-lived interaction
  biases instead of staying turn-local.
- Continuity and social grounding now persist across turns, so repeated
  interaction leaves slow identity traces instead of disappearing after one
  response.
- Those slow traces are now stored as `identity_trace`, `relationship_trace`,
  `community_profile_trace`, `context_shift_trace`, and `working_memory_trace`
  records, so the next turn can restore more than one transient gate state.
- Reinterpretation can now explicitly defer new meaning under unstable field
  conditions, keeping recall nearer to grounded observation until roughness
  settles.
- Slow field estimates now feed persistence and personality, so
  roughness/defensiveness can linger as continuity loss, strain, caution, and
  reduced exploration instead of affecting only one response gate.
- A first reusable `working_memory_core` now exists, so focus, unresolved
  loops, pending meaning, and carryover load are no longer only implicit in
  long memory or gate state.
- A first legacy bridge now exists for working memory too, so daily/nightly
  outputs can expose `inner_os_working_memory_snapshot/v1` without forcing a
  fake full migration of L1/L2/L3.
- That working-memory bridge now also lands in legacy diary entries as
  `working_memory_summary`, so a thin long-lived artifact exists even before a
  true episodic promotion rule is in place.
- A first promotion hook now exists too, so high-readiness working-memory
  traces can lift a few recent raw experiences into daily distillation
  candidates instead of staying diary-only.

## In Progress

- Sleep-like consolidation now has a reusable planner and a legacy bridge, but
  the real nightly loop still lives mostly in the original terrain/nightly
  system.
- Runtime logic still contains physiology-related behavior that should move
  further into reusable cores.
- `inner_os` now has a standalone HTTP app and router, but it still lives in
  the same repo and has not been packaged independently.
- Memory is reusable and now has typed classes plus runtime bridge helpers, but
  richer recall semantics and deeper consolidation are still thinner than the
  long-term target.
- Working memory now exists as a reusable short-horizon layer, but promotion
  into autobiographical and nightly structures is still thin.
- The current working-memory bridge is still a sidecar summary, not yet a
  true legacy-memory promotion path.
- Context-sensitive recall now distinguishes between observed terrain
  roughness and latent field-estimated roughness, but the long-horizon
  affective field is still lighter than the final target.
- The 2D surface is runtime-backed, but still remains a minimal life-surface
  rather than a mature world front-end.
- Sensor channels now reach runtime and living-world state, but only part of
  that signal feeds back into reusable `inner_os` cores.

## Not Yet Done

- A separately packaged distribution for the standalone HTTP service outside
  this repo.
- A published packaging workflow beyond the current export scaffold.
- Automated boundary checks for non-stdlib third-party requirements beyond the
  current import-scope test.
- Reference integration samples for existing AI-tuber or chat systems outside
  this repo.
- Clear package boundaries for extracting `inner_os` independently from
  `emot_terrain_lab`.
- Stable multi-memory class APIs for `observed_real`, `reconstructed`,
  `verified`, `experienced_sim`, and `transferred_learning`.
- Full promotion rules from working memory and nightly maintenance into
  autobiographical continuity.

## Practical Reading

As of now, the project is no longer only an application.
It has become a partial component OS.

The missing step is not basic architecture. The missing step is stronger
separation plus deeper lifecycle migration, so other systems can adopt the same
inner loop without depending on the full runtime.
