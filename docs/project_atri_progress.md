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

- Fixed the current architectural baseline in `docs/core/eq_core_v2_operating_concept.md` as `Core / Invariants / Regression Candidates`, so future work can be judged against the causal order `h -> I -> Π_q -> C -> P` instead of drifting toward planner-first or shell-first logic.
- Closed `shared / fallback / none` qualia-hint responsibility inside `inner_os/expression/hint_bridge.py`, keeping `shared` as the canonical same-turn path, `fallback` as bridge-only compatibility, and `none` as explicit neutral state.
- Kept `response_planner` in a downstream role: it now uses shared `qualia_planner_view` rather than serving as an independent felt reconstruction path.

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
- Context shift is now stored as `context_shift_trace`, so world/context mismatch is not consumed only inside one turn but can be restored on the next pre-turn and recall path.
- Terrain roughness is now split into observed roughness and latent field-estimated roughness inside recall payloads, so raw instability and slowly lingering instability are no longer conflated.
- Reinterpretation now has a `grounding_deferral` path, so when the field is rough, defensive, and socially weak, the OS prefers grounded observation over creating new reconstructed meaning too quickly.
- That grounding deferral now also reaches response gating, so a recalled “do not settle meaning yet” posture can reduce surface intensity and push conscious access toward clarification.
- Slow field estimates now feed `PersistenceCore` and `PersonalityIndexCore`, so roughness and defensive residue can survive beyond one gate decision and slightly alter continuity, strain, caution, affiliation, and exploration on later turns.
- Slow field residue now also feeds `DevelopmentCore` and is preserved into `community_profile_trace`, so long roughness/defensiveness can start shaping belonging, trust, norm pressure, and communal profile memory instead of remaining purely momentary state.
- Reinterpretation now reads `community_profile_trace` more directly through `community_profile_pressure`, so longer communal and institutional patterning can change meaning assignment itself instead of influencing only recall ranking and gate behavior.
- `community_profile_pressure` now flows through normalized memory records, recall payloads, and shared schema contracts, so communal profile influence is part of the reusable OS boundary rather than an internal-only side value.
- Added an import-boundary test so `inner_os` stays free of direct `emot_terrain_lab`, `apps`, and `scripts` dependencies.
- Added `examples/run_inner_os_integration_example.py` to show how an existing LLM loop can insert `inner_os` before and after generation.
- Added `inner_os/memory_core.py` so hook-driven recall and post-turn writes can pass through a thin, append-only memory layer.
- Added `inner_os/terrain_core.py` so hook-side state can describe affective slopes, recovery basins, ignition potential, and attractors.
- Added `inner_os/simulation_transfer.py` so simulation episodes can be recorded as `experienced_sim` and only promoted back as abstract `transferred_learning`.
- Added `inner_os/policy_packet.py` and began treating the reusable core output as an interaction policy packet, so runtime and expression now share a pre-LLM action posture instead of relying only on after-the-fact surface shaping.
- Added `inner_os/expression/content_policy.py`, so runtime surface shaping can now switch the response body skeleton by policy mode (`repair`, `respectful_wait`, `shared_world_next_step`, `contain`, `reflect`) instead of only changing prefixes and pauses.
- Tightened policy selection so `attune`, `repair`, `respectful_wait`, and `shared_world_next_step` no longer collapse into one generic surface path quite as easily, and added direct example coverage for those four visible strategies.
- Lifted content policy from a single skeleton string into an ordered content sequence, so runtime and future LLM shells can share `open -> follow -> close` action order instead of only re-reading one flattened sentence body.
- Made runtime surface shaping sequence-aware, so opening pace / gaze return / certainty now apply most strongly to the opening segment while follow/close segments preserve the policy order instead of being flattened and reshaped as one block.
- Added `inner_os/scene_state.py` so place, privacy, social topology, task phase, temporal phase, norm pressure, safety, and environmental load can be named as one reusable scene boundary instead of leaking only as scattered heuristics.
- Added `inner_os/interaction_option_search.py`, so action-family activation and candidate emergence can now be modeled as relative field-weighted options (`attune / wait / repair / co_move / contain / reflect / clarify / withdraw`) instead of only one linear planner path.
- Added `docs/core/interaction_option_search_architecture.md` to define the `Heart Field / Action Search / Resonance Evaluator / Articulation Shell` concept alongside the more implementation-oriented `Constraint Field / Interaction Option Search / Trajectory Evaluator / Expression Shell` naming.
- Added `inner_os/affect_blend.py`, `inner_os/constraint_field.py`, and `inner_os/conscious_workspace.py`, so terrain / relation / live regulation can now form a mixed affect state, an admissibility field, and a small reportable-vs-withheld foreground workspace before policy selection.
- Added `inner_os/contact_field.py` and `inner_os/access_projection.py`, so the route from terrain/scene pressure into conscious foreground is no longer implicit: local contact points now rise first, then access-ready regions are shaped before the workspace decides what is reportable, withheld, or still actionable.
- Added `inner_os/contact_dynamics.py`, so contact points are no longer purely one-shot: re-entry, carryover, and protective hold can now slightly stabilize or guard a repeated contact before access projection and workspace ignition.
- Added `inner_os/access_dynamics.py`, so membrane-side access is no longer recomputed as a fully fresh slice every turn: access-ready regions can now keep short-horizon inertia, gating hysteresis, and protective filtering before workspace ignition.
- Added `docs/core/conversational_object_architecture.md`, fixing the next architectural step: `EQNet core` is now explicitly framed not as a response-strategy menu but as a kernel that builds conversational objects, applies operations to those objects, evaluates expected effects on the other person, and only then hands a structured brief to the articulation shell.

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

Inventory note:

- `docs/inner_os_migration_inventory.md` now documents that `inner_os/` is a
  partial migration layer, not yet a total replacement for the older
  terrain/nightly heart system.
- `inner_os/sleep_consolidation_core.py` now provides a first reusable nightly-facing planner so sleep-like consolidation can begin migrating as an explicit OS boundary rather than staying only inside terrain/nightly maintenance code.
- `emot_terrain_lab/sleep/inner_os_bridge.py` now wraps existing `rest_state`,
  field metrics, and nightly summary signals into
  `inner_os_sleep_consolidation_snapshot/v1`, so sleep-like migration can start
  from the existing lifecycle instead of reimplementing it in parallel.
- `emot_terrain_lab/scripts/run_daily.py` now emits
  `inner_os_sleep_snapshot.json` through that bridge, so the daily pipeline can
  expose a reusable sleep-consolidation snapshot without replacing the original
  daily consolidation path.
- `emot_terrain_lab/sleep/cycle.py` can now also emit an adjacent
  `*_inner_os_sleep.json` snapshot when a legacy system is present, so nightly
  consolidation has its first direct bridge into the reusable OS boundary.
- `ops/nightly.py` and `schema/nightly.v1.json` now accept
  `inner_os_sleep_snapshot_path` and `inner_os_sleep_mode`, so nightly report
  JSON can reference reusable inner-os sleep snapshots when upstream bridges
  provide them.
- `ops/nightly.py` now auto-detects an existing
  `data/state/inner_os_sleep_snapshot.json` (or an explicitly configured
  snapshot path) and injects its path/mode into nightly reports, so the report
  path is no longer a passive schema-only receiver.
- Added `docs/co_living_lifeform_checklist.md` so progress can be judged
  against the whole co-living lifeform target instead of one subsystem at a
  time.
- Added `inner_os/working_memory_core.py`, so the OS now has a reusable
  short-horizon layer for current focus, unresolved loops, pending meaning, and
  carryover load rather than relying only on long memory and gate state.
- Hook integration now restores and writes `working_memory_trace`, so short-term
  focus can survive across turns and slightly influence recall and response
  gating.
- Added `emot_terrain_lab/memory/inner_os_working_memory_bridge.py`, so the
  existing daily/nightly lifecycle can emit
  `inner_os_working_memory_snapshot/v1` without pretending that short-term
  traces are already fully promoted into legacy L1/L2/L3 memory.
- `emot_terrain_lab/scripts/run_daily.py`, `ops/nightly.py`, and
  `schema/nightly.v1.json` now carry that sidecar snapshot path/focus/readiness,
  so the whole-system checklist can track short-term-memory handoff alongside
  sleep instead of treating working memory as an inner-only feature.
- `emot_terrain_lab/terrain/diary.py` and `terrain/system.py` now keep a thin
  `working_memory_summary` inside daily diary entries, so the bridge reaches a
  legacy long-lived artifact instead of stopping only at sidecar JSON.
- `terrain/system.py` now also lets high-readiness `working_memory` traces lift
  a small number of recent raw experiences into daily distillation candidates,
  so short-term focus has its first direct effect on legacy episodic promotion.
- `terrain/memory.py` now preserves `working_memory_promotion` inside L2
  episodes, so the promotion hook leaves an explicit episodic trace instead of
  disappearing after candidate selection.
- `terrain/memory.py` now also preserves a `working_memory_signature` in L3
  semantic patterns, so weekly abstraction can retain the rough short-term
  focus that fed promotion instead of dropping it entirely after L2.
- `terrain/system.py` and `terrain/diary.py` now read that L3
  `working_memory_signature` back into daily diary narration, so semanticized
  short-term focus becomes a thin continuity cue instead of staying invisible
  after abstraction.
- Added `inner_os/conversational_objects.py`,
  `inner_os/object_operations.py`, and `inner_os/interaction_effects.py`, so
  the core can now keep a separate record of what part of the other person's
  talk is being handled, what is being done with it now, and what effect on the
  other person is being aimed for before surface wording is chosen.
- `ops/nightly.py` now derives a small `inner_os` replay-bias context from
  `inner_os_working_memory_snapshot/v1` and feeds it into legacy go-sc gating,
  so replay retention can slightly favor events aligned with the current
  focus/anchor without replacing the existing nightly scoring body.
- `schema/nightly.v1.json` and nightly JSON summaries now expose
  `inner_os_working_memory_replay_bias`, so that replay-bias bridge is visible
  as an explicit lifecycle artifact rather than as a silent score tweak.
- `terrain/diary.py` and `terrain/system.py` now read the latest
  `inner_os_working_memory_replay_bias` summary back into daily diary
  narration, so replay-biased carryover leaves a thin next-day trace instead
  of ending inside nightly scoring only.
- `terrain/system.py` now also lets that replay-bias cue slightly prioritize
  which recent L2 episodes are sent into weekly abstraction, so the same
  carryover can affect semanticization instead of stopping at diary text.
- `terrain/memory.py` now preserves a thin
  `working_memory_replay_signature` in L3 semantic patterns when weekly
  abstraction is driven by that carryover cue, so the replay-biased handoff can
  survive abstraction instead of disappearing immediately after episode
  selection.
- `terrain/recall.py` now gives a small preference to semantic patterns whose
  `working_memory_replay_signature` matches the current query, so that same
  carryover can influence later recall selection instead of living only in
  weekly semantic storage.
- `inner_os/memory_core.py` and `inner_os/integration_hooks.py` now also accept
  a thin working-memory replay signature during `memory_recall()`, so replay
  carryover can bias reusable OS recall contracts as well as legacy terrain
  recall without replacing the existing rerank body.
- `inner_os/post_turn_update()` now also feeds that replay-signature cue into
  reconstructed-memory consolidation priority and audit output, so replay
  carryover can influence what is retained after a turn, not only what is
  recalled before a turn.
- `emot_terrain_lab/memory/inner_os_working_memory_bridge.py` and
  `terrain/system.py` now also merge replay reinforcement from recent
  `inner_os` reconstructed records into the weekly-abstraction carryover cue,
  so OS-side post-turn reinforcement can begin to shape legacy `L2 -> L3`
  promotion instead of stopping at reconstructed memory only.
- `terrain/memory.py` now also preserves a thin replay-shaped
  `recurrence_weight` on `L3` semantic patterns, so the same carryover can
  influence not only semantic selection but also how strongly a recurring
  pattern settles after abstraction.
- `terrain/recall.py`, `terrain/system.py`, and `terrain/diary.py` now also
  read that `recurrence_weight`, so replay-shaped semantic settling can affect
  later recall preference and the next-day working-memory signature narration
  instead of remaining a silent L3-only value.
- `emot_terrain_lab/memory/inner_os_working_memory_bridge.py` and
  `terrain/system.py` now also derive a thin semantic seed from the latest `L3`
  working-memory signature plus replay carryover, and feed it back into daily
  working-memory summary and promotion-candidate selection, so semantic
  settling begins to influence the next day’s short-horizon focus instead of
  only flowing one-way into abstraction.
- `inner_os/working_memory_core.py` now also accepts that semantic seed
  through `current_state / previous_trace`, so semantic settling can begin to
  alter OS-side carryover load, pending meaning, and focus anchoring instead
  of remaining only in the legacy bridge layer.
- `inner_os/integration_hooks.py` now also accepts a legacy-side
  `working_memory_seed` through `pre_turn_update(..., local_context=...)`, so
  that semantic carryover can actually enter OS-side short-horizon state
  instead of remaining only a dormant capability in `working_memory_core`.
- `emot_terrain_lab/hub/runtime.py` now also forwards an upstream
  `working_memory_seed` from the runtime surface into `inner_os.pre_turn_update()`,
  so that ingress path is no longer only available in isolated tests and can
  enter the real turn-start lifecycle when supplied by the surface layer.
- `emot_terrain_lab/hub/runtime.py` can now also derive that
  `working_memory_seed` directly from legacy `eqnet_system` summaries (`L3`
  working-memory signature plus nightly replay summary), so the carryover path
  is no longer limited to manually seeded runtime surface state.
- `runtime.process_turn()` now also exposes the active semantic carryover seed
  in `inner_os` metrics/persona metadata, so the same continuity cue is no
  longer silent and can be tracked across turn-start state, response shaping,
  and later audit.
- `runtime._build_context_payload()` now also emits working-memory seed tags
  into conscious-context snapshots, so the same continuity cue can begin to
  appear in diary-like episode records instead of living only in turn metrics.
- `eqnet_core/memory/diary.py` now also extracts those `wm_seed_*` tags into an
  explicit `working_memory_seed` field, so conscious diary rows can retain the
  continuity cue as structured data instead of leaving it buried only in raw
  tag lists.
- `eqnet_core/memory/mosaic.py` now also preserves the same
  `working_memory_seed` field in conscious episode logs, so replay/analysis
  paths can read the continuity cue without reparsing raw context tags.
- `MemoryMosaic` can now summarize a recent `working_memory_seed`, and
  `runtime.process_turn()` can feed that small summary back into
  `inner_os.memory_recall()` as a conscious-memory replay cue, so continuity
  traces stored in conscious episode logs can begin to influence later recall
  selection instead of remaining audit-only.
- `inner_os.memory_recall()` can now also read
  `semantic_seed_focus / semantic_seed_anchor / semantic_seed_strength` and
  slightly prefer matching records, so continuity cues shaped by replay and
  semantic carryover can now affect reusable OS recall as well as legacy
  terrain recall.
- recurring `L3` patterns can now also carry a thin `long_term_theme`, and
  that theme now flows through next-day seed derivation, diary narration,
  runtime seed propagation, OS-side working-memory carryover, and
  `inner_os.memory_recall()` reranking, so long-horizon continuity starts to
  move as one shared loop rather than only as separate short-term cues.
- that same `long_term_theme` now also gives a small relief/support signal to
  `inner_os.response_gate()`, so long-horizon continuity no longer affects only
  recall and seed synthesis but also slightly changes how the next turn stands
  up.
- `runtime._inner_os_working_memory_seed()` can now also fall back to that
  recent conscious-memory seed when no explicit surface seed is present, so a
  continuity cue written into conscious episode logs can begin to shape the
  next turn's short-horizon state instead of only later recall ranking.
- `terrain/system.py` now also merges that recent conscious-memory seed into
  the daily/nightly replay-carryover summary path, so continuity traces written
  into conscious episode logs can influence lifecycle-time continuity synthesis
  instead of remaining isolated to turn-time recall and seed fallback only.
- when that conscious-memory seed agrees with the current replay-carryover
  focus/anchor, the bridge now slightly reinforces carryover strength, so the
  conscious-memory cue can affect later daily/weekly selection weight instead
  of remaining only a visible auxiliary field.
- `terrain/memory.py` now also carries that reinforced continuity cue into
  `L3` recurrence weighting, so matching conscious-memory carryover can affect
  how strongly a semantic pattern settles after weekly abstraction instead of
  influencing only episode selection.
- `emot_terrain_lab/memory/inner_os_working_memory_bridge.py` now also lets
  that conscious-supported carryover slightly increase next-day
  `semantic_seed_strength`, so continuity traces that survive into `L3`
  recurrence can begin to re-enter short-horizon seed synthesis instead of
  ending only at semantic storage.
- `terrain/system.py` and `terrain/diary.py` now also surface that
  `semantic_seed_strength` inside the readable `Working memory signature`
  narration, so the next-day diary text and the next-day short-horizon seed
  share the same continuity cue instead of drifting into separate vocabularies.

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

## Latest Increment

- `Inner OS` の `ForegroundState` が candidate ranking だけでなく
  `reportability_scores` と `memory_candidates` を持つようになり、
  Value/Access の同じ選別過程から「何を話しやすいか」「何を残し
  やすいか」を一緒に出せるようになった。
- `select_foreground()` は continuity・danger・terrain-energy を使って
  reportability と memory-fixation 候補を補助評価するようになり、
  Emotional DFT の地形スナップショットが前景化だけでなく固定候補
  にも薄く効き始めた。
- `render_response()` は raw observation を見ないまま、
  foreground payload に reportability/memory 情報を載せられるように
  なったので、LLM を表出層に留めたまま access の理由を運べる。

- `long_term_theme` is now recorded as a dedicated daily/nightly summary instead
  of living only inside `working_memory_signature`.
- daily diary keeps `long_term_theme_summary` as structured data and renders a
  separate `Long-term theme:` line, so the day-level record can point at a
  continuing theme directly.
- nightly JSON can now expose `inner_os_long_term_theme_summary`, sourced from
  the working-memory snapshot rather than from ad-hoc tags.
- working-memory snapshots now preserve the latest
  `long_term_theme_focus/anchor/strength/kind/summary`, which gives daily and
  nightly reporting the same vocabulary for long-horizon continuity.
- runtime can now also fall back to `inner_os_long_term_theme_summary` when
  deriving the next turn's working-memory seed, so the day-level long-term
  theme is no longer just reported; it can re-enter the next turn's short-horizon
  state when stronger sources are absent.
- `inner_os` now keeps `long_term_theme_summary` through working-memory traces
  and passes it into recall reranking, so the long-term theme can bias not only
  the next turn seed but also which memories are slightly easier to retrieve.
- the same `long_term_theme_summary` now also adds a small reinforcement to
  post-turn reconsolidation priority, so long-term theme affects not only
  retrieval but also what is slightly more likely to stay.
- reconstructed `long_term_theme_reinforcement` now also flows back into legacy
  replay carryover and `L3 recurrence_weight`, so the OS-side long-term theme is
  starting to influence which semantic patterns become a little more recurring.
- conscious diary and `MemoryMosaic` now persist `long_term_theme` as structured
  fields extracted from `context_tags`, rather than keeping long-horizon
  continuity only in working-memory seed tags.
- runtime can now use `MemoryMosaic.latest_long_term_theme()` as a fallback
  source for the next turn's seed, so a long-term theme remembered on the
  conscious side can re-enter `pre_turn` even without an explicit world seed.
- `inner_os.memory.build_episodic_candidates()` now turns foreground-level
  `memory_candidates` into typed `EpisodicRecord` candidates, so Value/Access
  selection no longer ends only in payload fields and can begin to feed actual
  memory formation contracts.
- those episodic candidates now retain `salience` and `fixation_reasons`,
  which means the same selection process can explain not only what was seen,
  but why it was considered worth keeping.
- `inner_os.memory.build_memory_context()` now lifts those episodic candidates
  into lightweight semantic hints and continuity threads, so one turn can now
  carry a thin "what may remain meaningful next" bundle instead of only raw
  foreground and memory-candidate lists.
- `render_response()` can now include that memory context without exposing raw
  observation, so LLM-facing expression can stay downstream while still seeing
  which episodes and semantic hints are likely to matter.
- `inner_os.memory.build_memory_appends()` now converts that `MemoryContext`
  into append-ready `observed_real` / `reconstructed` records, so the
  foreground-to-memory path can actually enter `MemoryCore` instead of stopping
  at analysis-only hints.
- `IntegrationHooks.post_turn_update()` now reads those append-ready memory
  candidates as a weak continuity/meaning signal, so selected foreground no
  longer only gets written but can slightly bias `continuity_score`,
  `pending_meaning`, `social_grounding`, and the resulting identity /
  working-memory traces.
- the post-turn audit now carries `candidate_continuity_pull`,
  `candidate_meaning_pull`, and `candidate_social_pull`, which makes it
  possible to track whether foreground-level memory candidates are actually
  changing the next self-state instead of only being archived.
- foreground-selected episodic candidates can now retain `related_person_id`,
  and that person-target signal is carried through append-ready records into
  post-turn `identity_trace` / `relationship_trace`, so the system can begin
  to distinguish "what remained" from "who this remained with" instead of
  treating social carryover as anonymous.
- `pre_turn_update()` can now read person-specific `relationship_trace` records
  by `relational_world.person_id`, so partner-targeted carryover no longer
  stops at storage and can begin to alter the next entry stance for the same
  person.
- `memory_recall()` now passes `related_person_id` through biasing and payload,
  and `response_gate()` now gives a small relief when the current state already
  carries a person-specific relation trace, so the same partner can become a
  little easier to recall and approach within the turn itself.
- `MemoryContext` now keeps partner-targeted relation bias, `build_memory_appends()`
  can raise consolidation priority slightly for person-specific social episodes,
  and `render_response()` can shift to a lighter `check_in` stance when the turn
  is clearly about the same partner rather than an anonymous report.
- nightly reporting can now emit an `inner_os_partner_relation_summary` from
  recent `relationship_trace` records, so partner-targeted carryover is no
  longer trapped inside turn-time traces and can be surfaced as a reusable
  next-day summary beside the existing long-term theme summary.
- `_inner_os_working_memory_seed()` can now absorb that nightly partner summary
  as relation seed fields (`related_person_id`, attachment/familiarity/trust),
  and `pre_turn_update()` can recover the counterpart from `working_memory_seed`
  itself, so the same partner can begin to bias the next day's entry stance
  even when no explicit world-time person tag is present yet.
- partner-targeted relation seed now also enters `MemoryOrchestrationCore`
  and the working-memory promotion bridge, so the same partner can begin to
  bias not just next-day stance but also which episodes are promoted or
  replayed, instead of overnight relation carryover stopping at pre-turn mood.
- `SleepConsolidationCore` now reads partner-targeted relation seed
  (`related_person_id`, attachment/familiarity/trust, relation strength) and
  lets it raise replay, reconsolidation, autobiographical pull, and identity
  preservation bias, so same-partner carryover now reaches the overnight
  consolidation planner rather than ending at generic memory orchestration.
- `build_inner_os_sleep_snapshot()` now lifts
  `inner_os_partner_relation_summary` into the reusable sleep `current_state`,
  so nightly partner traces can survive into morning restoration through the
  same reusable `inner_os` boundary instead of requiring an ad hoc legacy path.
- person-targeted `relation_seed_summary` can now survive into `pre_turn`
  state, bias `memory_recall()` reranking, and appear in the reusable
  `response_planner` payload as a partner semantic summary, so the same
  partner's overnight carryover is starting to shape not only recall
  probability but also the morning conversational soft-start.
- `post_turn_update()` can now emit a reusable `person_registry_snapshot`
  for the current partner, and `pre_turn_update()` can absorb that snapshot
  back into the same partner's adaptive state before nightly runs, so
  partner-specific recognition and trust/familiarity growth no longer have to
  wait for nightly consolidation to begin updating.
- `person_registry_snapshot` can now be rehydrated into `PersonRegistry`, and
  `select_foreground()` now reads partner adaptive traits
  (attachment/familiarity/trust/continuity) as a `partner-trace` bias, so the
  same partner can begin to alter what rises into foreground instead of only
  affecting recall and stance after the foreground is already chosen.
- grounding/world access can now preserve `person_id_hint` from observed
  entities into `social_relation_graph`, and foreground selection can resolve
  that observation-time link back to the canonical partner node, so partner
  bias can start at the moment of re-recognition instead of waiting until a
  later recall or nightly merge.
- foreground partner bias is now also modulated by community/culture/role
  markers plus affiliation/caution timing gates, so the same person does not
  always influence stance equally: partner carryover can now strengthen,
  delay, or soften depending on communal alignment, cultural resonance, and
  the current social/personality axis rather than assuming a single fixed
  intimacy rule.
- grounding now carries the same partner/community context into affordance and
  symbol grounding: person-aware entities can emit `engage` affordances with
  familiar/respectful/delayed timing hints, and symbol grounding can carry
  partner-facing address/stance tags, so the same recognized person can start
  altering not only salience but also "how to approach" and "how to name"
  from the grounding stage.
- those grounding-stage `address_hint / timing_hint / stance_hint` values can
  now survive into `MemoryContext`, `response_planner`, and append-ready memory
  records as partner-specific soft-start cues and `social_interpretation`, so
  approach timing, naming stance, and what gets written down are beginning to
  align on the same person-aware axis instead of drifting apart downstream.
- partner grounding hints and `social_interpretation` can now be lifted into
  nightly relation summaries, survive the overnight bridge, and reappear in
  next-morning seed/state payloads, so yesterday's way of approaching and
  naming the same partner can begin to shape the next day's soft opening
  rather than resetting to a generic relation thread.
- next-morning `partner_social_interpretation` now also feeds
  `memory_recall()` reranking and `response_gate()` distance tuning, so the
  remembered approach style is beginning to affect not only the morning seed
  but also what comes back first and how softly or cautiously the system
  enters during the live turn itself.
- `partner_style_relief / caution` now also modulates foreground weighting and
  `response_planner` utterance stance selection, so the same remembered
  approach style can begin to affect moment-to-moment salience and whether the
  live entry feels warm, gentle, or measured instead of only biasing hidden
  recall/gate internals.
- partner-aware `utterance_stance` and relation episode naming now share the
  same helper, so how the system speaks to the same person and how it names
  that relation episode in memory are beginning to stay aligned.
- nonverbal partner expression now also derives from the same relation/situation
  axis, so gaze, pause, distance, and silence style begin to change with mood,
  timing, and remembered partner style instead of staying fixed.
- partner interaction no longer treats `love mode` as a single warmth scalar;
  it now carries relational mood axes such as future pull, reverence,
  innocence, care, and shared-world pull into nonverbal expression and memory
  naming.
- relational mood axes now also feed live-turn `response_gate()` and
  foreground weighting, so future-pull / care / reverence / shared-world
  effects can begin to shape ongoing salience and distance regulation instead
  of remaining planner-only annotations.
- `pre_turn_update()` now predicts relation mood, nonverbal expectation, and
  initial distance/hesitation tone before the live turn starts, so partner
  recognition can bias the opening stance from the first moment of contact.
- `post_turn_update()` now compares predicted opening nonverbal cues against
  observed gaze/pause/proximity/shared-attention signals and feeds that
  alignment or mismatch back into relation updates, so "how we expected this
  contact to go" can now slightly change trust, social grounding, strain, and
  later relationship traces instead of disappearing after the turn.
- `inner_os.interaction.trace` now derives a thin interaction trace from raw
  observation signals such as mutual attention, gaze-hold, aversion, pause
  latency, repair signal, and proximity delta, so post-turn relation updates
  no longer depend only on manually supplied `observed_*` tags.
- `inner_os.interaction.live_regulation` now derives past-loop pull,
  future-loop pull, fantasy-loop pull, strained-pause pressure, and distance
  expectation from relation mood plus current state, so remembered contact,
  anticipated next action, and meaning-overrun can begin to shape live
  nonverbal regulation instead of remaining only in memory or planner labels.
- `inner_os.interaction.orchestration` now provides a shared coordination layer
  that aggregates relation mood, nonverbal profile, live regulation, risk, and
  contact-readiness into one interaction snapshot, so planner and gate no
  longer drift as independent partial heuristics.
- `response_gate()` can now also advance a thin interaction stream state from
  raw observation cues, so shared attention, strained pause, repair-window
  pressure, and contact-readiness no longer stay fixed for the whole turn and
  can begin updating inside the live interaction itself.
- `response_planner` can now derive a thin `surface_profile` from the same
  shared interaction snapshot, so opening delay, response length, sentence
  temperature, pause insertion, and certainty style can begin shifting from
  live interaction state instead of staying as a fixed outer shell.
- runtime surface shaping can now also read that `surface_profile`, so
  turn-time output can begin to shorten, hesitate, soften, or stay more
  tentative from the same interaction stream instead of exposing only hidden
  control-state differences.
- `process_turn()` can now run a small two-step inner-os live loop, so the
  second response-gate pass can reopen stream state and surface shaping from
  the already-shaped draft instead of leaving the turn at one fixed reaction.
- that live loop can now run as a small multi-step turn-time regulation path,
  so stream state and surface shaping can be revised more than once within the
  same turn instead of only as one follow-up correction.
- that turn-time live loop can now stop on small stream-state deltas instead of
  always running a fixed count, so in-turn correction begins to behave more
  like adaptive convergence than like a rigid scripted number of passes.
- the interaction stream now keeps a short rolling window for shared attention
  and strained pause plus a small repair hold, so turn-time regulation starts
  carrying a little inertia instead of treating each step as memoryless.
- that rolling stream window now also feeds an `opening_pace_windowed` and a
  thin `return_gaze_expectation`, so the visible turn surface can begin to
  vary not only by stance but also by how attention and repair have been
  settling over the last few micro-steps.
- those opening/return expectations can now be compared against observed
  interaction-window signals inside `post_turn_update()`, so a mismatch in
  how the turn opened or how attention returned starts feeding back into
  relation strain and social grounding instead of staying as surface-only
  styling.
- the adaptive live loop can now also compare one step's opening/return style
  expectation against the next step's stream state and feed that mismatch back
  into strain, grounding, and contact-readiness, so mid-turn repair begins to
  happen before waiting for the end-of-turn update.
- that same live mismatch can now also reopen repair pressure and strengthen
  visible pause shaping inside the loop, so the turn surface begins to carry a
  little more mid-turn hesitation and repair instead of only changing at the
  end.
- the same mismatch can now also shorten response length and push certainty
  toward a more careful style when shared attention stays low, so mid-turn
  visible wording begins to tighten rather than only adding pauses.
- runtime now bridges `interaction_policy_packet` and ordered
  `content_sequence` into the LLM prompt itself, so the shell can begin from
  an `open/follow/close` plan instead of only inheriting after-the-fact
  surface shaping.
- Added `inner_os/resonance_evaluator.py` and wired it into both
  `response_planner` and `response_gate`, so candidate interaction options are
  now re-ranked not only by internal activation but also by an estimate of how
  each option is likely to affect the other person: whether it may increase the
  other person's burden, press them to explain too much, help them feel
  received, support their own speaking pace, and keep the next turn open.













