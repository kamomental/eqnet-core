# Module Contracts

## Grounding
`observe(input_streams) -> ObservationBundle`

## Affordance Engine
`infer_affordances(observation, world_state, self_state) -> AffordanceMap`

## Symbol Grounding
`ground_symbols(tokens, observation, affordances, value_state) -> SymbolGroundingMap`

## World Model
`update_world_state(prev_world_state, observation, affordances) -> WorldState`

## Self Model
`update_self_state(prev_self_state, world_state, events) -> SelfState`

## Continuity Layer
`update_person_registry(person_registry, observations, context) -> PersonRegistry`

`score_identity_continuity(person_node, observation) -> ContinuityUpdate`

## Value System
`compute_value_state(world_state, self_state, person_registry) -> ValueState`

## Access Layer
`select_foreground(world_state, self_state, value_state, attention_state, person_registry?) -> ForegroundState`

`ForegroundState` は candidate ranking, continuity focus, reportability scores,
memory fixation candidates を保持する

## Expression Layer
`render_response(foreground_state, dialogue_context) -> ResponsePlan`

## Memory Layer
`build_episodic_candidates(foreground_state, uncertainty, episode_prefix?) -> list[EpisodicRecord]`

`derive_semantic_hints(episodic_records, min_salience?) -> list[SemanticPattern]`

`build_memory_context(foreground_state, uncertainty, episode_prefix?) -> MemoryContext`

`build_memory_appends(memory_context) -> list[inner_os memory records]`

## Evaluation
`evaluate_run(trace) -> EvalReport`

## Guard rule
LLM bridge は `ForegroundState` だけを受け取る。
`ObservationBundle` や raw sensor stream を直接受け取ってはならない。
