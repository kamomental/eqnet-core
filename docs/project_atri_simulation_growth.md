# Project ATRI Simulation Growth Notes

Date: 2026-03-08

## Goal

Define how simulation-based growth should support Project ATRI without
overwriting the central design goal:

- real-world co-living and interaction stay primary
- simulation is used as a growth accelerator
- simulation experience must not be mixed naively with real experience

## Critical Position

Simulation is useful, but only if it remains subordinate to real-world
interaction.

If simulation becomes the main world, the system risks:

- overfitting to artificial social dynamics
- speaking with confidence that is unsupported in reality
- confusing imagined, simulated, and observed experience
- becoming impressive as a system but less natural as a companion

So the working position is:

1. Reality is the primary world.
2. Simulation is a secondary training world.
3. Transfer from simulation to reality must be selective, not automatic.

## Why Simulation Is Still Needed

There are practical reasons to keep a simulation layer:

- human video and embodied interaction data are still limited
- social situations are expensive to gather safely in reality
- long-horizon growth in reality is slow
- simulation can compress repeated practice into a shorter window

This makes simulation valuable as a controlled acceleration layer.

## Two-World Model

Project ATRI should be treated as a two-world lifeform:

### Real World

- five-sense grounding
- real interaction
- relationship continuity
- shared memory
- consent and safety constraints

### Simulation World

- accelerated social practice
- role trials
- counterfactual experience
- dense repeated episodes
- long-timescale growth rehearsal

### Transfer Layer

This layer decides what is carried back from simulation.

It should transfer:

- skills
- preferences
- policies
- conflict handling tendencies
- tolerance / curiosity / caution balance

It should not directly transfer:

- factual claims about reality
- concrete shared memories with real people
- real-world relationship history

## Memory Separation

Simulation requires explicit memory classes.

Recommended classes:

- `observed_real`
- `experienced_sim`
- `reconstructed`
- `verified_real`
- `transferred_learning`

### Meaning

- `observed_real`: what was actually perceived in reality
- `experienced_sim`: what was experienced inside simulation
- `reconstructed`: later reinterpretation or narrative integration
- `verified_real`: real-world facts later checked or confirmed
- `transferred_learning`: compact lessons promoted from simulation into policy or style

## Transfer Rules

Simulation should not write directly into real-world autobiographical memory.

Transfer should happen only through promotion rules such as:

1. repeated pattern observed across many sim episodes
2. stable benefit in decision quality
3. no conflict with real-world safety or consent
4. no contradiction with verified real experience

What gets promoted should be abstracted.

Good transfer:

- "pause and observe when signals conflict"
- "gentle clarification is better than forced certainty"
- "do not escalate quickly under ambiguity"

Bad transfer:

- "this person tends to reject me"
- "that place is dangerous"
- "this specific relationship already exists"

## Anti-Valley Principle

Simulation must not make the agent feel omniscient.

To avoid the uncanny valley:

- keep uncertainty visible
- keep transfer partial
- preserve hesitation and correction
- avoid instant mastery in the real world
- allow simulated learning to improve style and judgment gradually

The target is not "superhuman companion."
The target is "natural co-living being with compressed practice."

## Current Fit With Project ATRI

This direction fits the current architecture because:

- runtime is already the single state-owning individual
- memory layers are already separated in principle
- `inner replay` already supports future-facing internal simulation
- `observed_vision/v1` has introduced modality-aware memory handling

What is missing is an explicit simulation memory and transfer layer.

## Next Implementation Steps

### Step 1

Add simulation-aware memory labels and keep them separate from real-world logs.

Minimum new classes:

- `experienced_sim`
- `transferred_learning`

### Step 2

Create a promotion path from simulation outcomes to transferable policy hints.

This should produce:

- policy adjustments
- style tendencies
- caution / curiosity shifts

### Step 3

Expose provenance in runtime-facing internal data.

The runtime should know whether a cue came from:

- real observation
- simulation
- replay
- verification

### Step 4

Only after the above, design the actual simulation RPG loop.

Without this order, the simulation layer will likely blur worlds instead of
accelerating growth safely.

## Open Questions

- What kinds of simulation episodes are worth promoting?
- How should real users be protected from over-transferred simulated behavior?
- Should simulation transfer affect tone first, or policy first?
- How much of simulation history should remain inspectable?

## Recommended Immediate Direction

Do not build a large simulation world yet.

First implement:

1. simulation memory classes
2. transfer rules
3. provenance-aware recall

That keeps the project aligned with the real target:

an interactive, co-living, multi-sensory lifeform that grows naturally,
without becoming uncanny.
