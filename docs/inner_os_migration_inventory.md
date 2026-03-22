# Inner OS Migration Inventory

Date: 2026-03-15

## Purpose

This document is a corrective inventory.

`inner_os/` was built as a reusable receiver for the living-core logic, but it
has not yet replaced the existing `eqnet core` and `emot_terrain_lab` heart
systems wholesale.

So this document answers a narrower and more honest question:

- what already exists in the original system
- what has been mirrored or partially migrated into `inner_os`
- what is still duplicated
- what is still missing

It should be read before further migration work.

## Docs Reviewed

The following design documents were reviewed as part of this inventory:

- `docs/project_atri_core_architecture.md`
- `docs/project_atri_pipeline_os.md`
- `docs/project_atri_runtime_and_streaming.md`
- `docs/project_atri_simulation_growth.md`
- `docs/memory/README.md`
- `docs/index.md`

## Code Areas Reviewed

The following existing code areas were reviewed for this inventory:

- `emot_terrain_lab/terrain/system.py`
- `emot_terrain_lab/terrain/memory_palace.py`
- `ops/nightly.py`
- `emot_terrain_lab/scripts/run_daily.py`
- `emot_terrain_lab/scripts/run_weekly.py`
- `tests/e2e/test_e2e_hub_closed_loop.py`

## Main Finding

`inner_os/` is not yet a clean replacement for the existing core.

The repository currently has:

1. an older but deeper memory / nightly / replay / rest body
2. a newer and cleaner reusable `inner_os/` boundary

So the project is in a mixed state:

- `inner_os/` is ahead in portability and boundary quality
- the existing runtime / terrain / nightly system is still ahead in some long-run
  maintenance behaviors

That means current work should be understood as partial migration, not full
swap-over.

## Capability Mapping

### 1. Physiology / Cost / Rest

Existing system already has:

- fatigue thresholds
- rest mode triggers
- cooldown windows
- overload handling
- dampening during fatigue

Main files:

- `emot_terrain_lab/terrain/system.py`

`inner_os/` currently has:

- physiology core skeleton
- safety / hesitation / recovery summaries
- sensor-aware gating
- slow-field effects on persistence and personality

Main files:

- `inner_os/physiology.py`
- `inner_os/integration_hooks.py`
- `inner_os/persistence_core.py`
- `inner_os/personality_core.py`

Inventory judgment:

- original system is still stronger in actual rest machinery
- `inner_os/` is stronger in reusable interface shape
- migration is incomplete

### 2. Memory Storage / Distillation / Abstraction

Existing system already has:

- L1 raw memory
- L2 episodic distillation
- L3 semantic abstraction
- daily consolidation
- weekly abstraction
- diary generation

Main files:

- `emot_terrain_lab/terrain/system.py`

`inner_os/` currently has:

- typed memory classes
- append-only memory store
- recall payloads
- identity / relationship / community / context traces
- reconstructed memory generation
- simulation-origin memory separation

Main files:

- `inner_os/memory_records.py`
- `inner_os/memory_core.py`
- `inner_os/memory_bridge.py`
- `inner_os/reinterpretation_core.py`
- `inner_os/simulation_transfer.py`

Inventory judgment:

- original system is still stronger in explicit day/week memory lifecycle
- `inner_os/` is stronger in normalized record classes and reusable contracts
- they are not yet unified

### 3. Sleep-like / Nightly / Defrag-like Behavior

Existing system already has:

- nightly orchestration
- audit artifacts
- memory thermo / defrag metrics
- replay-related telemetry
- budget throttling and phase outputs

Main files:

- `ops/nightly.py`
- `tests/e2e/test_e2e_hub_closed_loop.py`

Existing terrain side also has:

- daily consolidation
- weekly abstraction
- rest state persistence

Main files:

- `emot_terrain_lab/terrain/system.py`

`inner_os/` currently has:

- forgetting heuristics
- reuse / interference / consolidation priority summaries
- replay horizon summaries
- context-sensitive recall allocation
- a reusable sleep-consolidation planner boundary

Main files:

- `inner_os/forgetting_core.py`
- `inner_os/memory_orchestration_core.py`
- `inner_os/sleep_consolidation_core.py`

The repository now also has a migration bridge:

- `emot_terrain_lab/sleep/inner_os_bridge.py`
- `emot_terrain_lab/scripts/run_daily.py`
- `emot_terrain_lab/sleep/cycle.py`

This bridge wraps legacy `rest_state`, field metrics, and nightly summary
signals into the reusable sleep planner without making `inner_os/` depend on
repo-specific lifecycle code.

The first live integration point is now the daily pipeline, which writes
`inner_os_sleep_snapshot.json` after `daily_consolidation()`.
Nightly replay consolidation can also emit an adjacent inner-os sleep snapshot
when a legacy system object is available.
The nightly report schema can now carry that snapshot path and mode, but the
full `ops/nightly.py` generation path has not yet been rewired to produce it on
its own.

Inventory judgment:

- `inner_os/` does not yet own a real sleep-like consolidation loop
- current `inner_os/` has support signals, not the full overnight engine
- original nightly/terrain paths remain essential

### 4. Affective Terrain

Existing system already has:

- terrain
- field
- membrane
- narrative projection
- memory palace positioning

Main files:

- `emot_terrain_lab/terrain/system.py`
- `emot_terrain_lab/terrain/memory_palace.py`

`inner_os/` currently has:

- terrain attractors
- transition roughness
- slow-field residue
- grounding deferral
- terrain-sensitive recall ranking

Main files:

- `inner_os/terrain_core.py`
- `inner_os/field_estimator_core.py`
- `inner_os/reinterpretation_core.py`
- `inner_os/memory_core.py`

Inventory judgment:

- `inner_os/` now has a meaningful public terrain skeleton
- original terrain system is still deeper and broader
- migration is partial, not complete

### 5. Replay / Inner Replay / Simulation

Existing system already has:

- replay-related telemetry and scripts
- inner replay references
- imagery replay hooks
- nightly replay summaries

Main files:

- `emot_terrain_lab/ops/task_fastpath.py`
- `emot_terrain_lab/ops/qualia_imagery.py`
- `ops/nightly.py`
- replay-related tests and docs

`inner_os/` currently has:

- `experienced_sim`
- `transferred_learning`
- selective promotion rules
- simulation-origin learning into development updates

Main files:

- `inner_os/simulation_transfer.py`
- `inner_os/development_core.py`

Inventory judgment:

- `inner_os/` has the cleaner simulation memory contract
- existing system still carries more replay ecosystem weight
- unified replay-to-growth migration is still incomplete

### 6. Conscious Access / Response Gating

Existing system already has:

- route logic
- talk mode logic
- gating and uncertainty shaping

Main files:

- `emot_terrain_lab/hub/runtime.py`

`inner_os/` currently has:

- `response_gate`
- schema-ed hook contract
- grounding deferral to gate
- clarify / soften / hesitation routing

Main files:

- `inner_os/conscious_access.py`
- `inner_os/integration_hooks.py`
- `inner_os/service.py`

Inventory judgment:

- this is one of the stronger migrated areas
- `inner_os/` is already usable here
- original runtime still owns some final integration details

## What Has Actually Been Migrated Well

The following are genuinely in good shape inside `inner_os/`:

- hook contracts
- service / HTTP / standalone transport
- normalized memory classes
- recall payload contracts
- relation / continuity / communal traces
- reinterpretation and reconstructed memory generation
- response gate as reusable boundary
- working-memory state as a reusable hook-side layer

## What Is Still Mostly Parallel

The following are not yet truly replaced; they exist in both worlds:

- physiology and rest handling
- nightly / defrag / sleep-like maintenance
- long memory lifecycle
- replay ecosystem
- deep affect-field mechanics
- short-term to long-term promotion, which now has a sidecar bridge but not a
  true legacy-memory handoff
- diary-level handoff now exists for working-memory summary, but episodic and
  autobiographical promotion rules are still missing
- a first daily distillation hook now exists, but it is still a narrow bridge,
  not a full short-term-to-L2 promotion model

## What Is Still Missing From Inner OS

For `inner_os/` to become a real replacement, not just a receiver, it still
needs:

- an explicit sleep-like consolidation core
- a real working-memory layer
- autobiographical promotion rules
- stronger predictive / mismatch machinery
- a clearer merger path with L1/L2/L3 and nightly operations

## Migration Rule Going Forward

From this point, new work should follow this order:

1. check whether a capability already exists in `emot_terrain_lab` or `eqnet`
2. decide whether `inner_os/` should wrap it, absorb it, or leave it where it is
3. only then add new `inner_os/` logic

This avoids deepening the current parallel-core problem.

## Short Summary

The current state is not:

- "fully migrated to `inner_os`"

The current state is:

- "`inner_os` is now a strong reusable skeleton"
- "the original terrain/nightly core still carries major lifecycle behavior"
- "the next step is systematic migration, not more blind expansion"
