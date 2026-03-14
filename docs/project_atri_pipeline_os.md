# Project ATRI Pipeline OS

Date: 2026-03-08

## Purpose

This document reframes Project ATRI as an injectable pipeline OS.

The goal is not to force users into one UI, one world renderer, or one
character shell.
The goal is to provide reusable inner-life pipelines that can be inserted into
existing LLM, AI-tuber, simulation, game, or agent stacks.

## Design Position

Project ATRI should be treated as:
- a reusable inner-life core
- a set of hookable pipelines
- a state and event contract boundary
- a thin expression-agnostic operating layer

It should not be treated as:
- a single monolithic app
- one mandatory 2D world
- one mandatory front-end
- one mandatory LLM provider

## What Must Stay Common

The reusable common layer should carry the parts that are hard to recover by
connecting an LLM alone.

Those common parts are:
- physiology
- temporal weighting
- affective terrain
- conscious access
- structured memory
- relational world context
- thin audit

These should remain upstream of text generation and upstream of visual style.

## What Should Stay Swappable

These parts should remain adapter-driven and replaceable:
- LLM providers
- VLM providers
- TTS or voice systems
- streaming front-ends
- 2D world renderers
- Godot, web, or Live2D surfaces
- simulation worlds

This means Project ATRI should expose contracts, not force one renderer.

## Minimum Integration Hooks

To be injected into someone else's system, the OS should expose small hooks.

### 1. pre_turn_update

Input:
- user input
- sensor input
- local context

Output:
- updated physiology state
- updated temporal pressure
- surface interaction hints

### 2. memory_recall

Input:
- text cue
- visual cue
- world cue
- current state

Output:
- recall payload
- retrieval summary
- recall pressure / ignition hints

### 3. response_gate

Input:
- draft prompt or draft response
- current state
- safety / boundary signals

Output:
- talk mode
- hesitation / guard bias
- allowed surface intensity
- response route

### 4. post_turn_update

Input:
- input
- output
- current state
- memory write candidates

Output:
- next-state updates
- memory appends
- recovery pressure
- selective audit record

## Current Concrete Boundaries

The repository now already has several useful public-facing boundaries.

- `RuntimeTurnResult`
- `project_atri_2d_state/v1`
- `project_atri_2d_event/v1`
- `/vision-frame`
- `/project-atri/2d-state`
- `/project-atri/2d-event`
- `VisionMemoryStore`
- `SSESearchAdapter`
- `LMStudioVLMAdapter`

These are early forms of the pipeline OS boundary.

## Core Directory Role

The `inner_os/` directory should become the reusable receiver for
inner-life cores.

Current direction:
- `inner_os/physiology.py`
- `inner_os/temporal.py`
- `inner_os/conscious_access.py`
- `inner_os/relational_world.py`

Expected next additions:
- `inner_os/terrain_core.py`
- `inner_os/memory_core.py`
- `inner_os/simulation_transfer.py`
- `inner_os/integration_hooks.py`

## How Existing Systems Should Use It

An external system should be able to keep its own:
- UI
- world
- character art
- streaming setup
- prompting style
- LLM backend

and still insert Project ATRI through one of two paths.

### Embedded library path

For Python systems:
- import hookable cores directly
- call runtime or core methods in-process

### Local service path

For mixed stacks:
- call HTTP endpoints
- exchange JSON state and event payloads
- keep the existing front-end unchanged

### Minimal integration example

A concrete example now exists at:
- `examples/run_inner_os_integration_example.py`

It shows one existing LLM loop calling:
- `pre_turn_update`
- `memory_recall`
- `response_gate`
- local generation
- `post_turn_update`

without depending on the full runtime.

## Important Boundary Rule

Project ATRI should enrich existing systems.
It should not require users to discard their current implementation.

That means:
- add thickness to time
- add embodied cost
- add memory ignition
- add hesitation and recovery
- add relation and world-situated behavior

without taking ownership of the whole application.

## Current Rebuild Direction

The rebuild should therefore continue in this order:

1. keep extracting reusable cores into `inner_os/`
2. keep defining hook-friendly public contracts
3. keep runtime as the canonical owner of full-state integration
4. keep world renderers and UIs downstream and swappable
5. keep audit thin enough that interaction remains primary

## Short Summary

Project ATRI should become:
- reusable as a heart
- injectable as a pipeline
- portable across front-ends
- stronger than a plain LLM wrapper
- still grounded in interaction rather than dashboards








