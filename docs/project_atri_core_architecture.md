# Project ATRI Core Architecture

Date: 2026-03-08

## Purpose

This document unifies design elements that already exist across the project but
are still scattered across runtime code, memory code, simulation notes, and
interaction discussions.

The problem is not that the project lacks ideas.
The problem is that core ideas are distributed across many files and are not
yet expressed as one architecture.

This document defines that architecture.

## Core Position

Project ATRI is not a thin LLM wrapper.
It is intended to be a pipeline OS for a co-living lifeform.

The central value is not just linguistic fluency.
The central value is a living core that can:
- feel cost
- carry tension and recovery
- remember with continuity
- remain situated in relation, culture, and community
- interact naturally without falling into the uncanny valley

LLM-based expression remains important, but it is not the center.
It is one expression layer on top of the core.

## The Seven Layers

The architecture should be understood as seven interacting layers.

### 1. Physiology

This layer carries the minimum inner cost structure required for responsible
behavior.

It includes:
- heartbeat
- autonomic balance
- pain and stress
- boundary and veto
- recovery

Without this layer, the system can still imitate emotions in language, but it
cannot carry embodied cost strongly enough to support responsible decisions.

### 2. Temporal Weighting

This layer gives the system history-sensitive emotional behavior.

It includes:
- accumulation
- decay
- lingering affect
- re-ignition
- habituation
- recovery lag

This prevents the system from behaving as if every turn were independent.

### 3. Affective Terrain

This layer treats emotion as a field or terrain rather than a single label.

It includes:
- attractors
- barriers
- recovery basins
- danger slopes
- memory ignition points

This is necessary if the system is to have something closer to a real inner
landscape rather than language-conditioned roleplay.

### 4. Conscious Access

This layer does not mean full self-explanation.
It means access control over what becomes active, verbalized, or withheld.

It includes:
- attention
- exposure control
- hesitation
- veto
- inner / outer switching

This layer prevents total flattening of inner state into public language.

### 5. Memory

Memory must remain strong and structured.

It includes:
- observed
- episodic
- associative
- reconstructed
- verified
- experienced_sim
- transferred_learning

Memory is not only storage.
It is also cue-triggered re-living and re-interpretation under present context.

### 6. Relational World

This layer places the lifeform in a world of people, places, things, culture,
and community.

It includes:
- people
- objects
- places
- roles
- cultural constraints
- community norms
- shared spaces

Without this layer, the system may have inner state but still remain socially
weightless.

### 7. Expression

Expression is the visible or audible surface.
It includes:
- text
- voice
- 2D world rendering
- streaming mode
- UI surfaces

This layer must remain downstream of the other layers.
It is important, but it should not own canonical state.

## Layer Priority

The intended priority is:

1. physiology
2. memory
3. relation / culture / community
4. temporal weighting
5. conscious access
6. expression
7. audit

Audit still matters, but it should remain thin and supportive.
If audit becomes the dominant design pressure, the system will drift toward a
dashboard rather than a lifeform.

## Existing Project Pieces Mapped To Layers

The current repository already contains many pieces of this architecture.
They are simply not unified yet.

### Physiology

Current candidates:
- `emot_terrain_lab/hub/runtime.py`
- `eqnet/hub/streaming_sensor.py`
- `eqnet/hub/runtime_sensors.py`
- `emot_terrain_lab/terrain/risk.py`
- `emot_terrain_lab/mind/shadow_estimator.py`

Current status:
- partial implementation exists
- not yet separated as clean reusable cores

### Temporal Weighting

Current candidates:
- runtime future-risk and future-hope logic
- replay traces
- recurrence counters
- memory hint pressure and cooldown logic

Current status:
- several mechanisms already exist
- they are still embedded in runtime instead of presented as one layer

### Affective Terrain

Current candidates:
- `emot_terrain_lab/terrain/*`
- `EmotionalMemorySystem`
- field / palace / graph structures

Current status:
- conceptually strong
- implementation is distributed
- still needs a cleaner public boundary

### Conscious Access

Current candidates:
- `TalkMode`
- `AccessGate`
- boundary signals
- response route logic
- memory hint disclosure policy

Current status:
- substantial logic already exists
- should be reframed as one access-control layer

### Memory

Current candidates:
- `emot_terrain_lab/memory/reference_helper.py`
- `emot_terrain_lab/memory/recall_policy.py`
- `emot_terrain_lab/hub/recall_engine.py`
- `emot_terrain_lab/terrain/memory_palace.py`
- `emot_terrain_lab/memory/vision_memory_store.py`
- `emot_terrain_lab/rag/sse_search.py`

Current status:
- strong base already exists
- simulation-aware memory classes are still missing

### Relational World

Current candidates:
- culture model and behavior mod logic
- place / partner / object fields in runtime logging
- world type and transition docs
- replay world assets
- mini world simulation

Current status:
- scattered but real
- needs to be promoted into an explicit layer

### Expression

Current candidates:
- `LLMHub`
- streaming / Gradio bridges
- observer pages
- living-world front-end
- 2D state and event contract

Current status:
- actively evolving
- should remain swappable and not become the state owner

## Pipeline OS Interpretation

Project ATRI should be treated as a pipeline OS, not a monolith.

The target shape is:

- Physiology Pipeline
- Temporal Pipeline
- Terrain Pipeline
- Conscious Access Pipeline
- Memory Pipeline
- Relational World Pipeline
- Expression Pipeline
- Thin Audit Pipeline

Each pipeline should expose small public contracts instead of leaking internal
implementation details.

## Minimum Public Contracts

The project already has the beginning of these contracts.
They should become the default integration surface.

### Turn Contract

- `RuntimeTurnResult`

### 2D State Contract

- `project_atri_2d_state/v1`

### 2D Event Contract

- `project_atri_2d_event/v1`

### Vision Contract

- `perception_summary`
- `observed_vision/v1`

### Recall Contract

- `retrieval_summary`
- memory cue text

## Why This Matters For Reuse

If these contracts remain stable, then other developers can reuse parts of the
system without adopting the whole repository.

That means they can take:
- memory only
- physiology only
- runtime full loop
- 2D state output only
- event ingress only

This is what makes the system a pipeline OS rather than a closed app.

## What Is Still Missing

The following pieces are still missing or incomplete.

### Missing explicit cores

These should become their own reusable modules:
- `HeartbeatCore`
- `AutonomicCore`
- `PainStressCore`
- `RecoveryCore`
- `BoundaryCore`

### Missing simulation memory separation

These still need concrete stores and promotion logic:
- `experienced_sim`
- `transferred_learning`

### Missing runtime exports

The new 2D state boundary exists, but other layers still need cleaner exports,
especially for physiology and world relation.

### Missing duplicate control

`observed_vision` can still become repetitive enough to distort SSE recall.

## Recommended Next Implementation Order

1. Extract or wrap physiology-related logic into explicit reusable cores.
2. Add simulation memory classes and selective transfer rules.
3. Expand runtime state export to expose world / relation / physiology more clearly.
4. Connect the living-world front-end to real runtime state instead of demo-only state.
5. Keep audit thin and secondary.

## Current Summary

The project already contains much of the desired system.
The issue is not absence but lack of integration.

The right next step is not a total rewrite.
The right next step is to unify scattered mechanisms into a coherent pipeline
architecture and then expose stable public contracts around it.

