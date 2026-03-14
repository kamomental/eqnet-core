# Component OS Status

This document tracks the current state of the reusable component OS view.

## Complete

- `inner_os/` exists as a root-level reusable receiver.
- Core modules exist for physiology, temporal weighting, conscious access, relational world, memory, terrain, and simulation transfer.
- Explicit hook contracts exist for `pre_turn_update`, `memory_recall`, `response_gate`, and `post_turn_update`.
- `InnerOSService` now exposes those hooks through a plain mapping-to-dict boundary.
- `EmotionalHubRuntime.process_turn()` already uses the hook path internally.
- `EmotionalHubRuntime.process_turn()` can now accept simulation-origin `transferred_lessons` and feed them into post-turn development updates through `inner_os`.
- Vision, SSE recall, and 2D state/event surfaces are already connected to runtime.
- Runtime-side memory references and observed-vision entries now bridge into `inner_os` memory classes through one OS-side helper path.
- Recalled memory can now be reinterpreted and reconsolidated inside `inner_os`, instead of leaving memory reframing implicit in runtime-only behavior.
- Environment pressure now feeds the same OS path, so scarcity, hazard, ritual, and institutional pressure can bias development and memory reinterpretation without relying on UI-only summaries.
- Repeated relationship signals now feed OS-side personality sediment, so attachment and familiarity can accumulate into longer-lived interaction biases instead of staying turn-local.
- Continuity and social grounding now persist across turns, so repeated interaction leaves slow identity traces instead of disappearing after one response.
- Those slow traces are now stored as `identity_trace` records, so the next turn can restore them even when live state is thin.
- Relationship sediment is now also stored per community/role/place scope, giving repeated interaction a more specific profile than one global continuity trace.

## In Progress

- Runtime logic still contains physiology-related behavior that should move further into reusable cores.
- `inner_os` now has a standalone HTTP app and router, but it still lives in the same repo and has not been packaged independently.
- Memory is reusable and now has typed classes plus runtime bridge helpers, but richer recall semantics and deeper consolidation are still thinner than the long-term target.
- The 2D surface is runtime-backed, but still remains a minimal life-surface rather than a mature world front-end.
- Sensor channels now reach runtime and living-world state, but only part of that signal feeds back into reusable `inner_os` cores.

## Not Yet Done

- A separately packaged distribution for the standalone HTTP service outside this repo.
- A published packaging workflow beyond the current export scaffold.
- Automated boundary checks for non-stdlib third-party requirements beyond the current import-scope test.
- Reference integration samples for existing AI-tuber or chat systems outside this repo.
- Clear package boundaries for extracting `inner_os` independently from `emot_terrain_lab`.
- Stable multi-memory class APIs for `observed_real`, `reconstructed`, `verified`, `experienced_sim`, and `transferred_learning`.

## Practical Reading

As of now, the project is no longer only an application.
It has become a partial component OS.

The missing step is not basic architecture. The missing step is stronger separation so other systems can adopt the same inner loop without depending on the full runtime.
