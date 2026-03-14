# Project ATRI 2D Simulation Interface

## Purpose

Define the minimum contract for a 2D simulation layer that can support Project
ATRI without replacing the real world as the primary place of growth.

This document is not a UI mock and not a game design document.
It fixes the boundary between:
- runtime as the living core
- simulation as a controlled acceleration layer
- 2D rendering as one expression surface

## Why This Exists

Project ATRI already has seeds for simulation and world state:
- `emot_terrain_lab/sim/mini_world.py`
- `docs/world_types.md`
- `docs/world_transition_rules.md`
- `docs/replay/world/harbor_town.json`

Those pieces are useful, but they are not yet a stable interface for a living
companion that moves between reality and simulation.

The main risk is drift into one of two bad outcomes:
- simulation becomes the main world and reality becomes secondary
- the 2D layer becomes a dashboard instead of an interaction surface

## Core Position

1. Reality remains the primary world.
2. Simulation is a growth accelerator, not a replacement life.
3. The 2D layer must expose life activity, not just audit state.
4. The same individual must persist across reality mode, streaming mode, and
   simulation mode.
5. Transfer from simulation to reality must be selective.

## Non-Goals

This interface does not try to do the following yet:
- build a full RPG loop
- define all maps, quests, or combat systems
- make simulation memories equal to real autobiographical memories
- replace direct human interaction with internal sandbox play
- optimize for "more intelligence" at the cost of natural presence

## Design Principle

The 2D layer should feel like a visible world that the companion inhabits.
It should not behave like a control panel.

That means the contract should prioritize:
- presence
- movement
- relation
- mood
- recovery
- recall
- ongoing interaction

over:
- audit completeness
- developer convenience
- raw metric density

## Runtime Boundary

The runtime remains the owner of canonical state.
The 2D simulation layer is a client of runtime state plus a source of bounded
simulation events.

```text
real sensors / user input / streaming input
    -> EmotionalHubRuntime
    -> simulation projection contract
    -> 2D renderer / 2D world loop
    -> bounded events back to runtime
```

The 2D layer must not directly mutate long-term canonical memory.
It can emit events, proposals, and simulation episodes.

## Minimum State Contract

A 2D adapter should receive a single payload that is stable across UI styles.

```json
{
  "schema": "project_atri_2d_state/v1",
  "timestamp": "2026-03-08T12:00:00Z",
  "identity": {
    "entity_id": "atri_core_01",
    "mode": "reality|streaming|simulation",
    "talk_mode": "watch"
  },
  "world": {
    "world_id": "harbor_town",
    "world_type": "infrastructure",
    "zone_id": "market",
    "time_phase": "day",
    "weather": "clear"
  },
  "body": {
    "energy": 0.72,
    "stress": 0.24,
    "love": 0.44,
    "arousal": 0.31,
    "recovery_need": 0.18,
    "attention_density": 0.57
  },
  "activity": {
    "state": "attend",
    "target": "user",
    "intent": "listen",
    "route": "watch",
    "streaming": false,
    "replay_active": false,
    "recall_active": true
  },
  "social": {
    "nearby_entities": ["user", "guide"],
    "bond_strength": {"user": 0.62},
    "safety_bias": 0.21
  },
  "memory": {
    "dominant_anchor": "bakery",
    "recent_recall_ids": ["vision-1", "vision-2"],
    "perception_available": true,
    "retrieval_hit_count": 3
  },
  "simulation": {
    "enabled": true,
    "episode_id": "sim-ep-001",
    "transfer_pending": false,
    "world_source": "mini_world"
  }
}
```

## Activity States

The 2D layer should render a small set of life states instead of arbitrary app
states.

Recommended minimum set:
- `idle`
- `attend`
- `watch`
- `talk`
- `recall`
- `replay`
- `rest`
- `sync`
- `stream`
- `simulate`

These map more naturally to a co-living being than task-centric states such as
`loading`, `working`, or `error`.

## Event Contract Back To Runtime

The 2D side may send bounded events back.
Those events must be explicit and typed.

```json
{
  "schema": "project_atri_2d_event/v1",
  "timestamp": "2026-03-08T12:00:05Z",
  "source": "2d_world",
  "event_type": "zone_enter",
  "world_id": "harbor_town",
  "entity_id": "atri_core_01",
  "payload": {
    "zone_id": "market",
    "salient_entities": ["vendor_01", "stall_blue"],
    "tags": ["market", "shared_space"]
  }
}
```

Recommended event types:
- `zone_enter`
- `zone_leave`
- `npc_contact`
- `object_focus`
- `sim_episode_start`
- `sim_episode_end`
- `sim_transfer_candidate`
- `rest_enter`
- `rest_exit`
- `stream_stage_enter`
- `stream_stage_exit`

## Simulation Memory Boundary

Simulation needs separate memory classes.
The 2D layer may help generate them, but must not promote them directly into
real-world autobiographical memory.

Use these classes:
- `experienced_sim`
- `transferred_learning`

Never write raw 2D events into `observed_real`.
A promotion step is required.

## Selective Transfer Rule

The default transfer policy should be conservative.
Only compact lessons should move from simulation into runtime policy or style.

Examples that may transfer:
- safer turn-taking tendencies
- better recovery timing
- lower panic under repeated low-risk exposure
- stronger preference for mutual coordination

Examples that should not transfer as fact:
- named places in simulation treated as real places
- fictional people treated as real acquaintances
- simulation-only episodes treated as lived reality

## Reuse Of Existing Assets

Current project assets already support this direction:
- `mini_world.py` can act as a simulation episode generator
- `harbor_town.json` can act as a reference world schema
- `world_types.md` and `world_transition_rules.md` define world semantics
- runtime `talk_mode` and `response_route` can anchor visible expression

So the next step is not to build a new world model from scratch.
The next step is to adapt these pieces behind a stable contract.

## Critical Risks

1. Too much dashboard logic
The 2D layer starts showing metrics instead of life.

2. Too much simulation authority
Simulation begins to overwrite reality-centered growth.

3. Too much game logic
The system becomes a toy RPG instead of a companion lifeform.

4. Too much perfect control
The visible world loses fragility and becomes uncanny.

## Recommended Next Steps

1. Add a `project_atri_2d_state/v1` serializer near runtime.
2. Add `project_atri_2d_event/v1` ingestion with bounded event types.
3. Introduce `experienced_sim` and `transferred_learning` stores.
4. Reuse `mini_world` as the first simulation provider.
5. Only after those steps, redesign the visual 2D scene.

## Current Runtime Surface

The first transport path is now available through the unified vision bridge.

- `GET /project-atri/2d-state`
- `POST /project-atri/2d-event`

This keeps the front-end exchangeable:
- the current web living-world surface can use it
- a future Godot front-end can use the same boundary
- runtime ownership stays on the Python side
## Current Decision

Do not build the full 2D simulation front-end yet.
First stabilize the state contract and simulation memory boundary.
That order preserves the whole-system goal:
co-living interaction first, accelerated growth second, audit third.

