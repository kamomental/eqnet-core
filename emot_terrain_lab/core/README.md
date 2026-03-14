# Core Directory

This directory is the receiver for reusable inner-life cores.

Its purpose is to host components that can later be reused by:
- `EmotionalHubRuntime`
- external Python integrations
- HTTP or IPC wrapper services
- alternate surfaces such as web, Godot, or simulation front-ends

The intent is to move responsibility here gradually, not all at once.

## Current Modules

- `physiology.py`
  - heartbeat
  - pain / stress
  - recovery
  - boundary bias
- `temporal.py`
  - temporal pressure accumulation
  - decay
  - re-ignition support
- `conscious_access.py`
  - surface access and intent exposure mapping
- `relational_world.py`
  - mode, world, zone, and simulation/reality placement

## Expected Direction

This directory should continue to receive reusable logic for:
- affective terrain
- memory-facing ignition helpers
- simulation transfer rules
- integration hook definitions

The runtime should remain the canonical integrator.
The code in this directory should remain reusable and low-coupling.
