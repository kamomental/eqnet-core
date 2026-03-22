# EQ Core v2 Operating Concept

Date: 2026-03-17

## Core

このプロジェクトの核は、観測からしか立てられない自己のうち、
身体・価値・可観測性にしばられた部分を felt candidate として切り出し、
それを前景化と保護へつなぐことです。

正本の順序は次です。

```text
h -> I -> Π_q -> C -> P
```

- `h`: 観測契約
- `I`: 観測からの自己推定
- `Π_q`: クオリア膜
- `C`: 競合的な前景化 / access / workspace
- `P`: 保護ポテンシャル

planner / language / surface shaping は、この経路の下流にある
`その感じで動きや話し方を変えるところ` に留めます。

## Role Vocabulary

docs と実装では、以後この言い方で役割をそろえます。

- `感じ取るところ`
  - 観測を見て、今どんな感じかを決めるところ
- `共通言語にするところ`
  - 決まった感じを、ほかの部分も同じ意味で読める形にするところ
- `その感じで動きや話し方を変えるところ`
  - 上で決まった感じに合わせて、動きや話し方を変えるところ

## Invariants

以下は回帰判定の基準として扱います。

- `h -> I -> Π_q` を上流の正本とする
- `C` と `P` はその下流で felt candidate を前景化・保護へつなぎ、
  `h -> I -> Π_q` を飛ばしたり作り直したりしない
- `planner / language / surface shaping` は `その感じで動きや話し方を変えるところ` であり、
  感じを新しく決めるところではない
- `shared` が正本、`fallback` は互換、`none` は明示的中立とする
- bridge 外は `qualia_planner_view` を読むところであり、
  raw qualia を再解釈しない
- `q` は access / workspace / planner / protection / memory を
  実際に変えなければならない
- `C` と `P` を強化しても `h -> I -> Π_q` の順序を逆流させない

## Regression Candidates

Any change that does one of the following should be treated as a regression
candidate until proven otherwise.

- weakens or obscures `h` as a concrete observation contract
- produces felt state without going through `I`
- lets planner or shell layers regenerate felt state independently
- keeps `q` in payloads but removes its causal effect on downstream behavior
- leaks `shared / fallback / none` responsibilities outside the bridge

## Purpose

This document redefines `EQ core` using the whole repository, not only the
latest `inner_os` slice.

For the cross-disciplinary, human-ergonomics-centered framing that connects
neuroscience, cognitive science, psychology, philosophy, and Buddhist process
views, see:

- `docs/core/eq_core_human_ergonomics_framework.md`

For the architecture that defines `EQNet core` around conversational objects,
object operations, intended interpersonal effects, and articulation handoff,
see:

- `docs/core/conversational_object_architecture.md`

The goal is to keep the project aligned with its original `eqnet-core`
direction:

- not a thin LLM wrapper
- not a single emotional terrain engine
- not a UI or shell-first assistant
- not a dashboard with emotional labels

`EQ core` is the living control kernel that turns body cost, memory, relation,
world placement, and conscious access into action posture.

LLM remains important, but only as an articulation shell.

## Core Position

`EQ core` should be treated as the heart OS of the project.

It is the part that must still produce a meaningful stance even if the language
shell is replaced, simplified, or temporarily degraded.

That means its canonical output is not text.

Its canonical output is an interaction-ready control packet that can drive:

- wait / approach / repair / soothe / ask
- attention allocation
- pause and return timing
- disclosure depth
- nonverbal stance
- memory write pressure
- affordance weighting

## What EQ Core Is Not

`EQ core` is not:

- only `AffectiveTerrainCore`
- only `TalkMode`
- only `inner replay`
- only `inner_os`
- only `runtime`
- only `qualia`
- only language generation policy

Each of those is a real part of the repository, but none of them alone is the
core.

## The Four Kernels

`EQ core` should be defined as four tightly coupled kernels.

### 1. Homeostatic Kernel

This is the minimum body-cost layer.

It includes:

- heartbeat and arousal proxies
- pain / stress / defensive loading
- recovery and rest demand
- safety margin and veto
- boundary protection

Repository anchors:

- `inner_os/physiology.py`
- `eqnet/hub/runtime_sensors.py`
- `eqnet/hub/streaming_sensor.py`
- `emot_terrain_lab/terrain/risk.py`

Without this kernel, the system can imitate emotion but cannot carry cost.

### 2. Replay-Memory Kernel

This kernel gives history, future anticipation, and re-interpretation.

It includes:

- L1 / L2 / L3 memory
- episodic replay
- future replay
- imagery replay
- nightly consolidation
- reinterpretation and reconsolidation
- working-memory carryover

Repository anchors:

- `emot_terrain_lab/terrain/memory.py`
- `emot_terrain_lab/terrain/memory_palace.py`
- `inner_os/memory_core.py`
- `inner_os/reinterpretation_core.py`
- `inner_os/working_memory_core.py`
- `inner_os/sleep_consolidation_core.py`
- `ops/nightly.py`

Without this kernel, the system loses continuity and becomes turn-local.

### 3. Relational-World Kernel

This kernel places the lifeform in people, objects, places, culture, and
community.

It includes:

- partner continuity
- place / role / culture / community pressures
- object and affordance fields
- social repair and rupture sensitivity
- shared-world formation

Repository anchors:

- `inner_os/relational_world.py`
- `inner_os/relationship_core.py`
- `inner_os/environment_pressure_core.py`
- `inner_os/object_relation_core.py`
- `inner_os/peripersonal_core.py`
- `inner_os/world_model/`

Without this kernel, the system may feel internally rich but remain socially
weightless.

### 4. Access-and-Action Kernel

This kernel is the momentary selector that decides how the system will take
part in the situation.

It includes:

- conscious access
- foreground selection
- hesitation and veto
- route / mode / repair posture
- nonverbal regulation
- interaction orchestration

Repository anchors:

- `inner_os/conscious_access.py`
- `inner_os/access/`
- `inner_os/integration_hooks.py`
- `inner_os/interaction/`
- `eqnet_core/models/talk_mode.py`

This kernel is the bridge between inner life and external action.

## Emotional Terrain and Qualia in This Model

Emotional terrain and qualia remain central, but they should be understood as
parts of the above kernels, not as the whole system.

### Emotional Terrain

The terrain is the value field that shapes motion, danger slopes, recovery
basins, and ignition points.

It belongs mainly to the homeostatic and replay-memory kernels.

### Qualia / Conscious Access

Qualia is not treated as free-floating poetry.

It is the access-controlled portion of the internal field that becomes:

- foregrounded
- reportable
- acted upon
- remembered as significant

It belongs mainly to the access-and-action kernel.

For a more explicit engineering model of qualia access, see:

- `docs/core/qualia_contact_access_model.md`
- `docs/core/qualia_membrane_projection_model.md`
- `docs/core/conscious_workspace_model.md`

That document separates:

- local `contact point`
- foregrounded `access region`
- outward-facing `reportable slice`

so qualia membrane can be handled as a staged access model rather than as one
flat mystical surface.

The missing companion layer is the `conscious workspace`, which holds a limited
foreground, separates `reportable` vs `withheld` slices, and couples access to
action before articulation.

## Stable Latents and Dynamic State

The repository should support a stable latent bank without collapsing dynamic
state into embeddings.

The intended division is:

- stable latents: slowly changing semantic / relational / cultural anchors
- dynamic state: fast-changing arousal, strain, hesitation, contact readiness,
  repair, shared attention

The bridge looks like this:

```text
stable latent bank
    -> relation prior / memory prior / affordance prior / culture prior
    -> dynamic state update
    -> terrain deformation and access bias
    -> interaction policy packet
```

This means embeddings, if used, should act as priors and field distorters.

They should not replace the dynamics.

## Canonical Output: Interaction Policy Packet

`EQ core` should ideally emit a reusable policy object before any text is
generated.

Example fields:

- `dialogue_act`
- `contact_readiness`
- `repair_bias`
- `distance_strategy`
- `attention_target`
- `gaze_return_expectation`
- `opening_pace`
- `disclosure_depth`
- `memory_write_priority`
- `affordance_priority`
- `do_not_cross`

This packet should be valid even when:

- text is disabled
- voice is disabled
- the shell changes
- the world is rendered visually
- the system acts through movement or robot control

## LLM Position

The LLM is not the owner of state.

It is the articulation shell that receives:

- selected content
- policy packet
- surface shaping instructions

Its role is to convert those into:

- wording
- explanation
- dialogue
- tone

The LLM may refine the outer layer, but it should not be the place where the
main emotional and relational decision is made.

## Runtime, Core, Shell, Audit

To keep the architecture practical, the project should separate four concerns.

### Core

The reusable heart OS:

- homeostasis
- replay-memory
- relational world
- access-and-action

### Runtime

The integration layer that joins sensors, memory, nightly lifecycle, world
state, and shell invocation.

Current anchors:

- `emot_terrain_lab/hub/runtime.py`
- `start_hub.py`

### Shell

The visible surface:

- LLM
- voice
- UI
- 2D world
- streaming adapters

### Audit

The measurement and safety layer:

- nightly reports
- telemetry
- invariants
- rollback and contamination checks

Audit is necessary, but it must remain downstream of the core.

## Repository Mapping

The current repository already approximates this shape.

- `inner_os/` is the reusable heart extraction path
- `emot_terrain_lab/` is the integration-heavy laboratory and legacy bridge
- `eqnet/` and `eqnet_core/` still hold important runtime-native concepts such
  as `TalkMode`, voice shaping, and sensor/runtime contracts
- `ops/` and `schema/` carry lifecycle and audit visibility
- `runtime/` and `apps/` expose the external operational surfaces

The correct move is not to erase the older project shape.

The correct move is to make `EQ core` the stable conceptual center across those
areas.

## Evaluation

`EQ core` should not be judged primarily by fluency.

It should be judged by whether it changes outcomes in ways that match a
co-living lifeform.

Primary evaluation axes:

- does the system change stance when the same words arrive from a different
  partner or context
- does cost change action mode before text
- does replay alter the next approach rather than only the next narration
- does relation history reshape repair, waiting, and return
- can the shell be replaced without destroying the core

## Immediate Design Implication

Near-term implementation work should prefer:

1. stronger action-policy planning before wording
2. stable latent bank to dynamic-prior bridge
3. affordance and relation priors feeding terrain and access
4. shell simplification, not shell inflation
5. outcome-based evaluation alongside internal-state metrics

## One-Sentence Definition

`EQ core` is the reusable heart OS that transforms body cost, replay-memory,
relational world placement, affective terrain, and conscious access into a
situated action posture, from which language is only one possible expression.
