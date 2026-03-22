# Co-Living Lifeform Checklist

Date: 2026-03-15

## Purpose

This document is the whole-system checklist for the project as a
co-living / co-resonant lifeform.

It should be used before making local progress claims about one subsystem.
Sleep, replay, VLM, `inner_os`, and runtime are all only parts of this list.

## Reading Rule

Each row below should be read in three columns:

- `existing core`: what already lives in `emot_terrain_lab` / `eqnet`
- `inner_os`: what has been wrapped or migrated
- `gap`: what is still missing for the lifeform goal

## Checklist

### 1. Physiology / cost / rest

- existing core:
  - fatigue, rest, overload, damping, nightly maintenance
- inner_os:
  - physiology, persistence, field residue, sleep planner boundary
- gap:
  - reusable cost model is still thinner than the legacy rest body

### 2. Temporal continuity

- existing core:
  - L1/L2/L3 lifecycle, daily and weekly consolidation
- inner_os:
  - temporal weighting, continuity, social grounding, lingering strain
- gap:
  - long-range temporal shaping is still split between both sides

### 3. Affective terrain

- existing core:
  - field, memory palace, danger/healing terrain, replay ecology
- inner_os:
  - terrain attractors, roughness, defensive residue, grounding deferral
- gap:
  - public terrain is usable, but still lighter than the original terrain body

### 4. Conscious access

- existing core:
  - route logic, TalkMode, runtime gating
- inner_os:
  - response gate, hesitation, soften/clarify, exposure control
  - explicit `SceneState` now names place/privacy/topology/phase/norm/safety as one reusable scene boundary instead of leaving those pressures scattered across gate heuristics
  - experimental `Interaction Option Search` now lets multiple action families stand up from the current field before final selection, instead of assuming one linear planner path
- gap:
  - final integration still partly runtime-owned
  - scene-aware option search exists as a boundary and concept, but it is not yet the canonical runtime selector

### 5. Short-term / working memory

- existing core:
  - L1 moment traces exist, but are not yet exposed as one reusable short-horizon layer
- inner_os:
  - a reusable `working_memory_core` now tracks focus, unresolved loops, pending meaning, and carryover load
  - daily / nightly can now reference `inner_os_working_memory_snapshot/v1` as a sidecar bridge
  - daily diary entries can now retain that summary as a thin long-lived handoff
  - daily consolidation can now lift a small number of recent raw traces when working-memory promotion readiness is high
  - L2 episodes can now retain `working_memory_promotion` metadata when that bridge fires
  - weekly abstraction can now preserve a `working_memory_signature` in L3 patterns
  - daily diary narration can now read that L3 signature back as a thin next-day continuity cue
  - nightly replay gating can now apply a small working-memory replay bias, so events aligned with the current focus/anchor are slightly easier to retain without replacing legacy scoring
  - the latest nightly replay-bias summary can now flow back into daily diary narration as a thin next-day carryover cue
  - weekly abstraction can now prioritize recent episodes whose working-memory promotion matches that replay-bias cue, so the carryover can influence which L2 material becomes semanticized
  - L3 semantic patterns can now retain a thin `working_memory_replay_signature`, so replay-biased carryover is not lost immediately after weekly abstraction
  - terrain recall can now give a small preference to semantic patterns whose replay signature matches the current query, so the carryover can re-enter later recall selection
  - `inner_os.memory_recall()` can now also accept a thin replay-signature cue and slightly prefer matching records, so the same carryover is no longer limited to legacy terrain recall
  - `inner_os.post_turn_update()` can now feed that replay-signature cue back into reconstructed-memory consolidation priority, so carryover starts to affect not only recall but also what is more likely to stay
  - legacy weekly abstraction can now also merge replay reinforcement from recent `inner_os` reconstructed records, so post-turn carryover can begin to reach `L2 -> L3` promotion instead of ending only inside OS-side memory
  - `L3` semantic patterns can now retain a thin `recurrence_weight` shaped by replay carryover, so the same cue begins to affect not only which patterns are formed but how strongly they settle as recurring structure
  - legacy terrain recall and diary narration can now also read that `recurrence_weight`, so replay-shaped semantic settling begins to affect later recall preference and next-day readable continuity cues
  - legacy working-memory summary and daily promotion candidate selection can now derive a thin semantic seed from `L3` signature plus replay carryover, so semantic settling begins to feed back into the next day’s short-horizon focus instead of only flowing forward into abstraction
  - `inner_os.working_memory_core` can now also absorb that semantic seed through `current_state / previous_trace`, so semantic settling begins to influence OS-side carryover load and pending meaning instead of remaining only in the legacy bridge layer
  - `inner_os.pre_turn_update()` can now also ingest a `working_memory_seed` from `local_context`, so that semantic carryover can be injected into OS-side short-horizon state instead of remaining only a latent capability inside `working_memory_core`
  - `runtime.process_turn()` can now forward an upstream `working_memory_seed` into `inner_os.pre_turn_update()`, so the seed ingress is no longer test-only and can enter the real turn-start path when the runtime surface provides it
  - `runtime` can now also derive that `working_memory_seed` from legacy `L3` signature plus nightly replay summary through `eqnet_system`, so the short-term carryover path no longer depends only on manually seeded surface state
  - conscious diary rows can now also preserve that continuity cue as a structured `working_memory_seed` field, so the turn-start carryover path is no longer visible only in metrics and raw context tags
  - conscious episode logs can now preserve the same `working_memory_seed` field, so later replay/analysis paths have a structured continuity cue instead of needing to reparsed raw context tags
  - recent conscious-episode seeds can now be summarized and returned to `inner_os.memory_recall()` as a small replay cue, so continuity traces stored in conscious memory can begin to influence later recall selection
  - that recent conscious-memory seed can now also fall back into `runtime`'s next-turn `working_memory_seed` derivation, so continuity traces written into conscious memory can begin to shape later short-horizon state instead of remaining replay-only
  - daily / nightly continuity synthesis can now also merge the latest conscious-memory seed into the replay-carryover summary path, so the same continuity trace is no longer split between turn-time recall and lifecycle-time carryover synthesis
  - when that conscious-memory seed agrees with the existing replay-carryover focus/anchor, it now slightly reinforces carryover strength instead of remaining metadata only, so conscious continuity traces can begin to affect later daily/weekly selection weight as well as visibility
  - that reinforced carryover can now also slightly increase `L3` recurrence weight when weekly abstraction forms a matching semantic pattern, so conscious continuity traces begin to affect not only selection but also how strongly a recurring pattern settles
  - the same conscious-supported carryover now also slightly increases next-day semantic-seed strength when `L3` signatures are turned back into short-horizon seed input, so recurring continuity traces can begin to return into the next day’s focus synthesis instead of ending at semantic storage
  - daily diary narration can now also display that resulting seed strength alongside the recurring signature, so the next day’s readable self-record and the next day’s short-horizon seed share the same continuity cue instead of diverging into separate vocabularies
  - `inner_os.memory_recall()` can now also read `semantic_seed_focus / anchor / strength` and give a small preference to matching records, so the same continuity cue is no longer limited to legacy recall paths
  - recurring `L3` patterns can now expose a thin `long_term_theme`, and that theme now flows through next-day seed derivation, diary narration, runtime seed propagation, OS-side working-memory carryover, and `inner_os` recall reranking, so long-term continuity is starting to move as one shared loop instead of isolated short-term cues
- gap:
  - promotion from short-term focus into autobiographical structure is still weak
  - the bridge is still mostly a sidecar-plus-bias path, not a true self-narrative promotion rule

### 6. Long-term memory / reconsolidation

- existing core:
  - L1 raw, L2 episodic, L3 semantic, memory palace, diary, nightly abstraction
- inner_os:
  - normalized memory classes, recall payloads, reconstructed records, simulation memory separation
- gap:
  - the two worlds are still bridged, not yet unified

### 7. Relation / attachment / trust

- existing core:
  - recall helpers, social runtime signals, replay traces
- inner_os:
  - relationship sediment, identity trace, role/place/community traces
- gap:
  - repeated partner-specific long-horizon modeling is still thin

### 8. Culture / community / norms

- existing core:
  - culture docs, world docs, runtime logging, policy notes
- inner_os:
  - community profile trace, culture/community resonance, communal reinterpretation pressure
- gap:
  - norms do not yet fully behave like long-horizon action cost functions

### 9. Sleep / replay / overnight self-maintenance

- existing core:
  - daily consolidation, weekly abstraction, nightly report, replay telemetry
- inner_os:
  - sleep planner, bridge, daily output, nightly-adjacent snapshot, report reference path
- gap:
  - reusable sleep boundary exists, but the overnight engine is still owned by legacy core

### 10. Simulation / world separation

- existing core:
  - replay and simulation-related ecosystem is broader
- inner_os:
  - `experienced_sim`, `transferred_learning`, selective transfer
- gap:
  - world-separated growth rules remain partial

### 11. Interactive expression

- existing core:
  - runtime, TalkMode, VLM/SSE, streaming, Observer, living-world surface
- inner_os:
  - service, HTTP, standalone app, response gate boundary
  - policy-first interaction packets and content skeletons can now visibly split
    attune / repair / respectful-wait / shared-world responses instead of only
    decorating one generic reply body
  - those policy-first outputs now also preserve an ordered `open/follow/close`
    content sequence, so expression can begin to depend on action ordering
    rather than only one flattened sentence skeleton
- gap:
  - expression still depends on runtime composition more than on pure OS export

### 12. Self continuity / autobiographical growth

- existing core:
  - diary, memory palace, nightly summaries, long-lived memory assets
- inner_os:
  - identity trace, community profile trace, working-memory trace, reconstructed memory
- gap:
  - autobiographical promotion rules remain incomplete
  - day-level self-summary is still thin, but `long_term_theme` now survives as
    a dedicated daily/nightly summary instead of being hidden only in
    `working_memory_signature`
  - conscious diary / mosaic now also keep `long_term_theme` structurally, but
    this is still a continuity cue loop, not a full autobiographical narrative
  - `inner_os` can now let foreground-selected memory candidates bias
    post-turn continuity and working-memory traces, but this remains a thin
    self-carryover mechanism rather than a robust autobiographical narrator
  - foreground-selected memory candidates can now preserve `related_person_id`
    into post-turn traces, so self-carryover is starting to become
    person-specific rather than fully anonymous, but long-horizon
    relationship growth is still thin
  - person-specific `relationship_trace` can now be restored at `pre_turn`
    through `relational_world.person_id`, so the same partner can begin to
    bias entry stance and attachment/familiarity recovery instead of all
    social carryover being pooled
  - person-specific relation carryover now also reaches recall reranking and
    response gating, so partner-specific memory and softness can begin to
    appear inside the same turn rather than only between turns
  - partner-targeted relation bias now reaches response planning and memory
    priority, so "how to speak" and "what to keep" are starting to align on
    the same person axis instead of only sharing generic social carryover
  - nightly can now summarize the strongest recent `relationship_trace` into
    `inner_os_partner_relation_summary`, and runtime can feed that back as a
    relation seed on the next day, so partner carryover is starting to survive
    beyond the immediate turn loop
  - `pre_turn` can now recover `related_person_id` from `working_memory_seed`
    itself, which means a partner-specific overnight seed can still reopen
    attachment/familiarity/trust even before explicit perception has fully
    reidentified that person in the current turn
  - partner-specific overnight seed now also biases memory orchestration and
    working-memory promotion/weekly abstraction selection, so the same partner's
    episodes can begin to survive into replay and promotion instead of only
    softening the next entry stance
  - partner-specific relation seed now also enters `SleepConsolidationCore`
    replay/reconsolidation/autobiographical bias, and the sleep bridge can
    derive those fields directly from `inner_os_partner_relation_summary`, so
    same-partner carryover is starting to survive the overnight consolidation
    phase rather than stopping at next-day pre-turn mood
  - `relation_seed_summary` now survives into morning `pre_turn` state,
    `memory_recall()` reranking, and `response_planner` partner semantic hints,
    so the same partner's overnight thread can begin to alter both what comes
    back first and how the next interaction gently opens
  - turn-end now emits a reusable `person_registry_snapshot`, and next
    `pre_turn` can absorb that partner snapshot directly, so partner-specific
    recognition and trust/familiarity growth no longer depend only on nightly
    summarization to start thickening
  - rehydrated `PersonRegistry` now feeds partner adaptive traits into
    foreground selection as `partner-trace`, so the same partner can begin to
    change what gets noticed first, not only how recall or response softening
    behave after selection
  - observed entities can now carry `person_id_hint` into world-state
    `social_relation_graph`, and access selection can resolve that hint back to
    the canonical partner node, so partner-specific salience can begin at
    recognition time instead of only after later memory-stage recovery
  - partner-specific salience is now modulated by community/culture/role
    markers and by affiliation/caution timing gates, so influence is no longer
    treated as a flat "close person" rule: communal alignment can strengthen
    or delay foreground pull depending on culture and current social stance
  - grounding can now carry partner/community context into affordance and
    symbol grounding, so recognized people no longer only affect what rises
    into foreground: they can also alter approachability timing and address
    stance from the grounding stage itself
  - grounding-stage partner hints now survive into response planning and
    append-ready memory writes, so approach timing, address stance, and social
    interpretation start to remain aligned instead of being recomputed
    independently downstream
  - partner grounding hints now also survive into nightly relation summaries
    and next-morning seed/state restoration, so the remembered way of
    approaching the same person can persist across the overnight boundary
  - overnight partner `social_interpretation` now also feeds recall reranking
    and response-gate distance tuning, so remembered approach style begins to
    affect live-turn re-entry rather than only static morning preload
  - partner style relief/caution now also affects foreground weighting and
    actual utterance stance choice, so remembered distance style begins to
    shape visible live interaction rather than staying only in hidden control
    variables
  - partner-aware utterance stance and relation episode naming now use the
    same helper, so speaking style and relation-memory naming no longer drift
    apart as separate heuristics
  - nonverbal expression now also follows the same relation/situation axis,
    so gaze, pause, distance, and silence style can vary with mood and timing
  - relation handling now starts to split future-pull / reverence / innocence /
    care / shared-world pull instead of collapsing all affinity into one
    undifferentiated warm mode
  - relational mood now also touches response-gate and foreground weighting,
    so those axes begin to shape ongoing salience and distance regulation
  - pre-turn now predicts relation mood / nonverbal expectation / opening
    distance tone so partner recognition can bias the very first entry
  - post-turn now compares predicted opening cues against observed gaze /
    pause / proximity / shared-attention signals and lets that interaction
    alignment or mismatch feed back into relation traces, trust, grounding,
    and strain, so the system can begin to notice "we expected one kind of
    contact but received another"
  - raw nonverbal observation signals (mutual attention, gaze hold/aversion,
    pause latency, repair signal, proximity delta) can now be summarized as a
    reusable `interaction_trace`, so relation updates no longer require only
    hand-tagged `observed_*` labels to notice that shift
  - past-loop recall, future-directed pull, and fantasy/meaning overflow can
    now be summarized as live interaction regulation, so those inner loops are
    beginning to affect shared attention, strained pause, and distance
    expectation during the turn instead of staying only as hidden memory-state
  - planner and response-gate now share an explicit interaction orchestration
    snapshot, so relation mood, nonverbal regulation, live loop pressure, and
    contact readiness can start being coordinated as one whole instead of as
    loosely parallel partial heuristics
  - response-gate can now advance a thin interaction stream state from raw
    observation cues, so shared attention, strained pause, repair-window
    pressure, and contact readiness can begin shifting during the turn instead
    of staying frozen at one precomputed setting
  - response planning can now derive a thin `surface_profile` from the same
    interaction orchestration, so opening delay, response length, pause
    insertion, certainty style, and sentence temperature begin to move with
    the live interaction state instead of staying as a fixed shell over the
    turn
  - runtime surface shaping can now read that `surface_profile`, so visible
    turn output can begin to shorten, hesitate, soften, or stay tentative
    from the same interaction stream instead of keeping those differences only
    as hidden controls
  - `process_turn()` can now run a small two-step live loop, so the same turn
    no longer has to remain at one frozen interaction-gate state after the
    first surface pass
  - that live loop can now revise stream state and visible surface shaping
    more than once within the same turn, so turn-time reaction is starting to
    become iterative rather than single-correction only
  - turn-time live correction can now stop on small stream-state deltas, so
    the same turn is beginning to settle by adaptive convergence instead of a
    rigid fixed number of response passes
  - the interaction stream now keeps a short rolling window for shared
    attention / strained pause and a small repair hold, so turn-time
    regulation begins to carry a little inertia instead of behaving as a
    memoryless step-by-step loop
  - that short stream window now also informs visible opening pace and return
    gaze expectation, so the turn surface begins to reflect not only stance
    labels but also the recent settling pattern of attention and repair
  - opening pace and return-gaze expectation can now be checked against the
    observed interaction window after the turn, so mismatch in how contact
    opened or re-formed begins to feed relation strain and social grounding
    rather than remaining a surface-only styling choice
  - the adaptive live loop can now also react to opening/return mismatch
    between one micro-step and the next, so repair pressure and contact
    readiness begin shifting inside the turn instead of only after it ends
  - live opening/return mismatch can now also thicken pause insertion and
    reopen repair pressure inside the loop, so hesitation and repair start to
    show up as part of the visible turn surface rather than only as hidden
    regulation
  - when that mismatch persists under low shared attention, the visible turn
    can now also become shorter and more careful, so wording itself begins to
    tighten with the same live repair pressure
  - the runtime now sends `interaction_policy_packet` and ordered
    `content_sequence` into the LLM prompt bridge, so shell generation can
    start from a policy-first `open/follow/close` structure instead of only
    receiving post-hoc surface adjustments

## Current Priority Order

The next steps should follow this order:

1. strengthen short-term to long-term promotion
2. continue bridging legacy nightly/lifecycle into reusable boundaries
3. reduce duplicated memory semantics between legacy core and `inner_os`
4. deepen relation/culture effects on long-horizon development

## Important Constraint

This checklist is for the lifeform as a whole.

A local improvement, such as sleep export or replay telemetry, should not be
presented as whole-lifeform completion.
