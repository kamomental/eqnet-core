# Conscious Workspace Model

Date: 2026-03-18

## Why This Layer Exists

`qualia membrane` alone is not enough to justify `EQNet core` as a distinct
heart OS.

If the system only has:

- memory recall
- relation bias
- policy hints
- LLM-facing tone control

then it can collapse back into high-end middleware.

The missing layer is a consciousness-like workspace that can:

- hold a limited foreground
- keep some tension active but withheld
- remain re-readable across several steps
- feed action selection before language
- leave a trace for later replay and reinterpretation

This document defines that layer.

## Position In The Stack

```text
stimulus / body / memory / scene
    -> affective terrain response
    -> contact points
    -> qualia membrane / access shaping
    -> conscious workspace
    -> interaction option search
    -> policy packet
    -> articulation shell
```

The workspace is downstream of `qualia membrane` and upstream of
`interaction option search`.

In the current implementation, that upstream route is now made explicit as:

- `contact field`
- `access projection`
- `conscious workspace`

## Required Properties

The workspace is considered consciousness-like only if it provides:

1. limited capacity
2. persistence across several steps
3. selective reportability
4. selective withholding
5. re-entry from later evaluation and memory pressure
6. binding with self / relation / scene
7. coupling to action selection, not only language

Without these, it is only a feature buffer.

## Data Shape

The current minimal implementation uses:

- `WorkspaceSlot`
  - `slot_id`
  - `label`
  - `source`
  - `activation`
  - `reportable`
  - `withheld`
  - `binding_tags`

- `ConsciousWorkspace`
  - `workspace_mode`
  - `ignition_score`
  - `workspace_stability`
  - `recurrent_residue`
  - `dominant_slot`
  - `active_slots`
  - `reportable_slice`
  - `withheld_slice`
  - `cues`

This is intentionally small.

It is not claiming full consciousness.
It is implementing a reusable foreground-holding layer.

## Workspace Modes

- `preconscious`
  - contact points exist, but no stable foreground is formed
- `latent_foreground`
  - ignition has begun, but reportability is still unstable
- `foreground`
  - at least one slice is available for explicit policy and report
- `guarded_foreground`
  - something is active, but the system should not force it into report

`guarded_foreground` is especially important.

It prevents the design from collapsing into:
"if it is active, it must be spoken."

## Relation To Qualia Membrane

`qualia membrane` and `conscious workspace` are not the same layer.

- `qualia membrane`
  - compresses
  - integrates
  - shapes what can become access-ready
  - enforces ergonomic admissibility

- `conscious workspace`
  - holds a limited foreground
  - separates reportable vs withheld content
  - keeps residue across several steps
  - exposes bound content to action selection

Short form:

- membrane = access shaping
- workspace = foreground holding

## Relation To Fast / Mid / Slow Paths

- `Protective Path`
  - immediate protective action
  - boundary hold, overload suppression, fast veto

- `Conscious Workspace`
  - limited foreground and withholding
  - feeds midpath choice

- `Slow Path`
  - narration, autobiographical write, reconsolidation

The workspace is not a replacement for protective reactions.
It is the place where fast traces can later become meaningful and reusable.

## Relation To Action Selection

The workspace should influence:

- admissible interaction families
- disclosure depth
- repair vs wait vs co-move bias
- memory write priority
- do-not-cross constraints

This is why `workspace_mode`, `reportable_slice`, and `withheld_slice` are
fed into:

- `constraint_field`
- `interaction_policy_packet`
- `action_posture`
- runtime metadata

## Current Implementation Scope

Current scope is intentionally minimal:

- one ignition pass
- small slot set
- reportable / withheld split
- actionable slice for pre-linguistic policy coupling
- recurrent residue
- runtime carryover hook

Current scope does **not** yet provide:

- full recurrent ignition dynamics
- explicit slot competition over time
- workspace decay equations
- nightly workspace reconsolidation
- cross-turn self-binding beyond lightweight carryover

So this is a valid `v0`, not a completed consciousness architecture.

## Engineering Value

This layer matters because it makes `EQNet core` more than:

- prompt middleware
- relation-weighted retrieval
- emotional style routing

It creates a place where:

- something can be active
- not yet speakable
- still actionable
- still action-relevant
- still memory-relevant

That is the minimum bridge from terrain and qualia language toward a genuine
heart OS.

## Current Engineering Split

The current `v0` is implemented as:

- `qualia membrane`
  - access shaping before workspace
- `ignition loop`
  - foreground ignition, re-entry, recurrent residue
- `reportability gate`
  - `reportable / withheld / actionable` split
- `conscious workspace`
  - held foreground packet passed to policy selection

This keeps the model explicit:

- membrane is not the whole workspace
- withheld content is not discarded
- actionable content is not reduced to speakable content
