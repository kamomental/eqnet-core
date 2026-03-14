# Project ATRI Progress Snapshot

## What Is In Place

The current system already has a usable living-core path.

- `start_hub.py` acts as the canonical CLI entry point.
- `EmotionalHubRuntime` exposes a stable runtime-facing turn API.
- `RuntimeTurnResult` carries response, affect, perception, and retrieval.
- LM Studio vision can feed image summaries into runtime turns.
- SSE retrieval can recall related entries from local memory JSONL files.
- A unified vision bridge handles both webhook telemetry and frame-driven runtime turns.
- Streaming frame watching can call the same runtime path used by `/vision-frame`.
- `observed_vision/v1` is now a formal append-only memory layer.
- SSE models can be stored locally and are ignored by Git.

## Recently Completed

Recent work completed in this phase:

- Canonicalized `start_hub.py` as the runtime entry point.
- Added runtime-facing DTOs for turn results.
- Routed image input through LM Studio VLM into `perception_summary`.
- Routed visual cue text into SSE retrieval and `retrieval_summary`.
- Added `VisionMemoryStore` and moved vision logging into `observed_vision/v1`.
- Unified `scripts/vision_bridge.py` and `scripts/streaming_vlm_bridge.py` around one runtime path.
- Restored `EmotionalMemorySystem._apply_consent_filters()` so runtime shutdown can persist state again.
- Migrated `logs/vision_memory.jsonl` into the `observed_vision/v1` schema.
- Added retry support to the LM Studio VLM adapter.
- Switched SSE to local-first loading with Hugging Face fallback.
- Added the simulation growth position document and kept simulation subordinate to reality.
- Added the 2D simulation interface document so future world rendering can reuse runtime state without replacing the living core.
- Added a first living-world front-end route at `/living-world` that renders the harbor world as a life-surface instead of a dashboard.
- Added runtime-facing `project_atri_2d_state/v1` serialization and `project_atri_2d_event/v1` ingestion behind the vision bridge API.
- Wired the living-world surface to `/project-atri/2d-state` and `/project-atri/2d-event`, replacing local demo-state playback with real runtime polling and mode events.
- Added a `vision_sensor_frame/v1` contract so `/vision-frame` can accept sensor metadata and normalize it before `RuntimeSensors` ingestion.
- Extended `vision_sensor_frame/v1` so audio, body, and place channels can flow through the same `RuntimeSensors` entry as vision metadata.
- Added duplicate suppression in `VisionMemoryStore` so near-identical observed-vision records do not dominate SSE recall.
- Added a natural degraded-vision path so missing or unstable VLM perception becomes tentative presence instead of a hard technical surface error.
- Extended `project_atri_2d_state/v1` with a thin `sensing` layer so voice, body, and place cues can reach the living-world surface without turning it into a dashboard.
- Fed voice, body, and place cues back into `inner_os` pre-turn and response-gate decisions so sensing now changes route, intensity, and softening behavior rather than only surface text.
- Added a unified core architecture document to gather physiology, temporal weighting, terrain, memory, relation, and expression into one model.
- Added a first reusable physiology core module and routed runtime heart / stress / recovery / safety summary logic through it.
- Added `inner_os/` as the root-level receiver for reusable physiology, temporal, conscious-access, and relational-world modules, while leaving `emot_terrain_lab/core/` as a compatibility layer.
- Added a Project ATRI pipeline OS document to define injectable hooks for other existing systems.
- Added the first reusable `inner_os/integration_hooks.py` module with `pre_turn_update`, `memory_recall`, `response_gate`, and `post_turn_update` hook contracts.
- Added `inner_os/hook_contracts.py` so the hook boundary has explicit DTO-style request contracts.
- Added `inner_os/service.py` so existing systems can call the same hook set through a plain mapping-to-dict boundary.
- Added observer-side HTTP endpoints for the four `inner_os` hooks so external systems can begin integrating without importing runtime internals.
- Added an `inner_os` HTTP manifest so external integrations can discover the public hook surface without reading runtime internals.
- Added standalone `inner_os/http_router.py` and `inner_os/http_app.py` so the hook surface can run outside Observer.
- Added `scripts/export_inner_os_package.py` so the current `inner_os` package can be exported as a standalone scaffold with its own `pyproject.toml`.
- Added typed memory normalization for `observed_real`, `reconstructed`, `verified`, `experienced_sim`, and `transferred_learning`, and began feeding culture/community/place context through `inner_os`.
- Added `inner_os/memory_bridge.py` so existing runtime memory references and observed-vision entries can be bridged into normalized `inner_os` memory records.
- Fed `culture_id`, `community_id`, `social_role`, and `place_memory_anchor` from runtime surface/world context into `inner_os` pre-turn, recall, post-turn, and 2D-state paths.
- Began reducing double memory management by appending bridged runtime memory references and bridged observed-vision records into `inner_os` memory candidates during `process_turn()`.
- Aligned `reference_helper` recall results with `inner_os` memory vocabulary by emitting `record_kind`, `record_provenance`, `summary`, and `memory_anchor`, and mirrored that vocabulary into recall logging.
- Extended `inner_os` recall payloads so OS-side memory recall now returns `summary`, `text`, `record_kind`, `record_provenance`, and `source_episode_id` in the same family as `reference_helper`.
- Standardized service and HTTP outputs so hook results now carry explicit result schemas, and `memory_recall` advertises `inner_os_recall_payload/v1` through both the manifest and the runtime response body.
- Centralized public schema names and contract helpers into `inner_os/schemas.py`, so memory records, recall payloads, service responses, and the HTTP manifest now point at one shared contract source.
- Brought `hook_contracts.py` into the same schema family, so hook input DTOs now carry explicit input schemas instead of living as ad-hoc Python-only shapes.
- Added `inner_os/development_core.py` so culture, community, and social role now influence reusable pre-turn, response-gate, and post-turn behavior as small developmental pressures instead of staying as passive labels.
- Connected `transferred_learning` back into `DevelopmentCore`, so simulation-origin lessons can now alter belonging, trust, norm pressure, and role commitment instead of remaining memory-only.
- Added `inner_os/reinterpretation_core.py` and threaded `recall_payload` into post-turn updates, so recalled memory can be socially reframed and reconsolidated as `reconstructed` memory inside the OS layer.
- Extended `EmotionalHubRuntime.process_turn()` so simulation-origin `transferred_lessons` can enter post-turn growth through `inner_os`, not only through standalone service calls.
- Added `inner_os/environment_pressure_core.py` and fed resource, hazard, ritual, institutional, and social-density pressure back into developmental updates and memory reinterpretation, so environment now changes how recollection is re-read inside the OS.
- Added `inner_os/relationship_core.py` and `inner_os/personality_core.py`, so repeated social contact can sediment into attachment, familiarity, rupture sensitivity, and longer-lived caution/affiliation/reflective biases inside the OS.
- Added `inner_os/persistence_core.py`, so continuity, social grounding, and recent strain now survive across turns instead of living only inside one response step.
- Reinterpretation now reads persistence traces too, so recall is biased not only by immediate social pressure but also by yesterday-level continuity and lingering strain.
- `identity_trace` records are now written into `inner_os` memory and restored on the next pre-turn, so persistence is no longer only in live state.
- Community/role/place-scoped `relationship_trace` records are now also restored, so repeated contact in the same social setting can sediment into attachment and trust instead of staying generic.
- Community/culture-scoped `community_profile_trace` records are now restored and fed back into recall, reinterpretation, personality bias, and response gating, so repeated communal context can alter not only what is remembered but how readily the lifeform opens or hesitates.
- `DevelopmentCore` now reads attachment, familiarity, trust-memory, and role-alignment traces, so relationship sediment feeds back into belonging and trust bias instead of remaining parallel state only.
- `PersonalityIndexCore` now also reads continuity, social grounding, recent strain, trust-memory, and role-alignment, so repeated relation and persistence traces shape caution/affiliation/exploration more directly.
- `MemoryCore.build_recall_payload()` now does a small relational re-ranking, so recall selection is biased by community/role/place and current personality traces instead of staying a pure term match.
- Recall re-ranking now also reads environment pressure, so risky or rule-heavy situations can slightly bias which memory kinds come forward first.
- Added an import-boundary test so `inner_os` stays free of direct `emot_terrain_lab`, `apps`, and `scripts` dependencies.
- Added `examples/run_inner_os_integration_example.py` to show how an existing LLM loop can insert `inner_os` before and after generation.
- Added `inner_os/memory_core.py` so hook-driven recall and post-turn writes can pass through a thin, append-only memory layer.
- Added `inner_os/terrain_core.py` so hook-side state can describe affective slopes, recovery basins, ignition potential, and attractors.
- Added `inner_os/simulation_transfer.py` so simulation episodes can be recorded as `experienced_sim` and only promoted back as abstract `transferred_learning`.

## Current Operational Picture

The runtime path now looks like this:

```text
user input / image / streaming frame
    -> EmotionalHubRuntime
    -> VLM perception summary
    -> SSE recall cue + retrieval summary
    -> response + response_meta
    -> observed_vision memory append
```

The HTTP bridge path also works:

```text
sensor / frame webhook
    -> /vision-frame
    -> runtime turn
    -> perception + retrieval + output JSON
```

This means the project now has a real integration path for:
- perception
- lightweight recall
- interaction output
- append-only vision memory
- streaming ingress
- exchangeable 2D state and event boundaries
- bridged runtime memory records with explicit memory classes
- culture/community/place-aware recall and post-turn memory appends

## Whole-System Direction

The project should continue to be judged as a co-living lifeform, not as a
monitoring dashboard and not as a raw intelligence benchmark.

Current direction is fixed as follows:
- interaction remains primary
- audit remains supportive
- reality remains primary
- simulation remains subordinate
- transfer from simulation to reality must be selective
- natural presence matters more than perfect fluency

Reference:
- `docs/project_atri_core_architecture.md`
- `docs/project_atri_operating_model.md`
- `docs/project_atri_runtime_and_streaming.md`
- `docs/project_atri_simulation_growth.md`
- `docs/project_atri_2d_simulation_interface.md`

## Known Remaining Work

The next important items are not random feature additions.
They are the remaining boundaries needed to support the whole system.

High priority:
- Extract or wrap explicit physiology cores from existing runtime mechanisms.
- Continue collapsing duplicated runtime-side memory semantics into `inner_os` record classes.
- Deepen selective transfer and world-separation rules for simulation-origin material.

Medium priority:
- Expose `observed_vision` and simulation-related activity in Observer without turning it into a dashboard-first experience.
- Decide how streaming mode modifies expression without creating a second personality.
- Decide how nightly / weekly consolidation should treat simulation-origin material.

Lower priority for now:
- Large simulation RPG loops beyond the current renderer skeleton.
- Rich 2D front-end redesign.
- Broad world authoring.

## Risks To Watch

- Vision logs may become repetitive enough to distort recall quality.
- Simulation could become more vivid than reality and steal the growth loop.
- Observer could become more developed than interaction surfaces.
- The system could become smoother while losing fragility and feeling uncanny.

## Recommended Next Checks

To keep the project aligned with the whole-system goal, the next checks should be:

1. Continue splitting physiology into explicit reusable cores beyond the current minimal extraction.
2. Keep moving runtime-side memory references and vision traces toward `inner_os` memory classes.
3. Feed richer physiology and place cues into reusable cores instead of leaving them as surface-only summaries.
4. Tighten simulation memory transfer with strict world separation.
5. Evaluate interaction quality after each step, not just technical correctness.













