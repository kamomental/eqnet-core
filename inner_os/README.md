# Inner OS

This package is the reusable inner-life operating system layer.

It is intended to hold the parts that should remain portable across:
- existing LLM applications
- AI-tuber stacks
- streaming systems
- simulation worlds
- web or game front-ends

Its focus is not one character shell or one renderer.
Its focus is the shared inner stack:
- physiology
- temporal weighting
- working memory
- conscious access
- relational world placement
- memory and affective terrain
- same-turn protection, commitment, and agenda selection
- same-turn learning-mode meta-control and small social experiment selection
- matrix relation memory scaffolding for M²RNN-like outer-product memory heads
- overnight carry and transfer-safe continuity
- identity arc reconstruction for long-term self narrative
- identity arc registry carry for day-spanning self-story continuity
- relation arc reconstruction for person/group continuity
- relation arc registry carry for day-spanning relationship continuity
- group relation arc reconstruction for thread-level social continuity
- persona memory fragment selection for context-driven personality recall
- temporal memory evidence bundle scaffolding for time-aware recall sidecars
- temporal memory orchestration sidecar for time-aware recall and reentry contexts
- qualia membrane temporal bias bridge for time-aware field deformation before readout, weak same-turn policy shaping, and sleep/nightly carry observability
- boundary transform and residual reflection sidecars, so candidate expression/action flow can preserve the distinction between internal candidate pressure, boundary-side deformation, and what remains unsaid after guarded contact
- contact reflection state sidecars and dashboard observability, so guarded / open / absorbing contact can be inspected as a typed reflection mode instead of being inferred only from the final sentence
- daily-carry and dashboard observability for boundary/residual pressure, so guarded contact can be inspected not only by what was expressed but also by what stayed unsaid
- green-kernel contracts for projecting memory / affective / relational / boundary / residual flow into a shared inner field, so “Green” can become a replaceable mechanism contract instead of a prose-only metaphor
- recent dialogue state sidecars for short-turn continuity, so reopening / continuing / fresh-opening pressure can be carried as typed state instead of being left to prompt-only interpretation
- discussion-thread sidecars for short-turn issue continuity, so unresolved / revisiting / settling topic pressure can influence expression before it collapses into generic support language
- issue-state sidecars for turn-local issue phase, so exploring / pausing / resolving pressure can be carried as typed state before it falls back to prose-only interpretation
- discussion-thread registry snapshots for short-horizon issue carry, so repeated anchors can persist across turns instead of collapsing back into isolated replies
- nightly / transfer carry for discussion-thread registry summaries, so short-horizon unresolved anchors can survive sleep-like rollover and model swap without forcing them into long-term arc semantics
- autobiographical-thread summaries for weak promotion from discussion / issue / residual carry into working-memory pressure, so lingering unfinished topics can persist across turns and slightly bias later reopening without collapsing straight into a fixed self-story
- nightly / transfer carry for autobiographical-thread summaries, so weak unfinished-topic pressure can survive sleep-like rollover and model swap and still reappear as a small reopening / return-point bias instead of disappearing between sessions
- interaction constraints, repetition guard, and turn-delta sidecars for content-sequence shaping, so expression can preserve one context-specific difference without reusing the same generic reply line
- short Japanese conversational compaction for opening-line flows, so `opening + presence + return` can surface as a brief spoken response instead of a stacked support script
- live engagement state for streaming systems, so comment pickup, riffing, and small topic seeding are decided as typed same-turn state instead of renderer-only improvisation
- situation risk state and emergency posture sidecars, so the same object/person/contact can shift meaning across routine task, public exposure, private breach, or relationship rupture before dialogue vs distance/exit is decided
- sleep-like consolidation planning
- thin service access for existing systems through `InnerOSService`

`emot_terrain_lab` may keep adapters and integrations.
`inner_os` should remain the root-level reusable heart.

## Naming note

- Long-term structured memory retrieval/orchestration should use the descriptive name
  `Temporal Memory Orchestration`.
- Avoid adopting external paper acronyms such as `ASMR` as repo-level module names,
  because they collide with common non-technical meanings and weaken code readability.
- When a paper term must be mentioned, keep it in docs as a citation only, and map it to
  a repo term such as:
  - `temporal_memory_orchestration`
  - `TemporalMemoryOrchestrator`
  - `memory_evidence_bundle`
