# Core Docs

最小で見直しやすい core loop と、回帰判定の基準をまとめる入口です。

## Role Vocabulary

docs と実装では、以後この言い方で役割をそろえます。

- `感じ取るところ`
  - 観測を見て、今どんな感じかを決めるところ
- `共通言語にするところ`
  - 決まった感じを、ほかの部分も同じ意味で読める形にするところ
- `その感じで動きや話し方を変えるところ`
  - 上で決まった感じに合わせて、動きや話し方を変えるところ

## Current Regression Baseline

- `h -> I -> Π_q -> C -> P` を正本の因果順序とする
- `shared` qualia hints を正本、`fallback` を bridge 内互換、`none` を明示的中立とする
- planner / language / surface shaping は `その感じで動きや話し方を変えるところ` であり、
  感じを新しく決めるところではない

## Core Docs Map

- Core quickstart (2026-04-25): `docs/core/core_quickstart_2026_04_25.md`
- Conversation contract eval: `inner_os/evaluation/conversation_contract_eval.py`
- EQ core v2 operating concept: `docs/core/eq_core_v2_operating_concept.md`
- EQ core human ergonomics framework: `docs/core/eq_core_human_ergonomics_framework.md`
- EQ core reconstruction for risk reduction: `docs/core/eq_core_reconstruction_for_risk_reduction.md`
- Conversational object architecture: `docs/core/conversational_object_architecture.md`
- Interaction option search architecture: `docs/core/interaction_option_search_architecture.md`
- Qualia contact-access model: `docs/core/qualia_contact_access_model.md`
- Qualia membrane projection model: `docs/core/qualia_membrane_projection_model.md`
- Qualia structure + temporal memory integration: `docs/core/qualia_structure_temporal_integration.md`
- Qualia structure state: `docs/core/qualia_structure_state.md`
- Heartbeat structure state: `docs/core/heartbeat_structure_state.md`
- Utterance reason chain: `docs/core/utterance_reason_chain.md`
- Listener action state: `docs/core/listener_action_state.md`
- EQNet boundary / residual layers: `docs/core/eqnet_boundary_residual_layers.md`
- Emergency expression bridge: `docs/core/emergency_expression_bridge.md`
- Deep disclosure reflection: `docs/core/deep_disclosure_reflection.md`
- Contact reflection state: `docs/core/contact_reflection_state.md`
- Green kernel contracts: `docs/core/green_kernel_contracts.md`
- Surface expression selection: `docs/core/surface_expression_selection.md`
- Surface context packet: `docs/core/surface_context_packet.md`
- llm_hub response cause: `docs/core/llm_hub_response_cause.md`
- LLM bridge contract: `docs/core/llm_bridge_contract.md`
- Discourse shape: `docs/core/discourse_shape.md`
- Discourse shape reasoning: `docs/core/discourse_shape_reasoning.md`
- Codex use-case environment: `docs/core/codex_usecase_environment.md`
- Codex plugin root: `docs/core/codex_plugin_root.md`
- Route surface priority: `docs/core/route_surface_priority.md`
- Evaluation criteria: `docs/core/evaluation_criteria.md`
- Evaluation operating policy: `docs/core/evaluation_operating_policy.md`
- Evaluation snapshot (2026-03-28): `docs/core/evaluation_snapshot_2026_03_28.md`
- Evaluation targets: `docs/core/evaluation_targets.md`
- Evaluation loop (anchor progress, 2026-03-28): `docs/core/evaluation_loop_anchor_progress_2026_03_28.md`
- Mechanism issue map: `docs/core/mechanism_issue_map.md`
- Amadeus transition issue map: `docs/core/amadeus_transition_issue_map.md`
- Amadeus transition progress (2026-03-29): `docs/core/amadeus_transition_progress_2026_03_29.md`
- Amadeus transition progress (2026-03-31): `docs/core/amadeus_transition_progress_2026_03_31.md`
- Repo fragment integration map: `docs/core/repo_fragment_integration_map.md`
- Growth state: `docs/core/growth_state.md`
- Development transition policy: `docs/core/development_transition_policy.md`
- Epistemic state: `docs/core/epistemic_state.md`
- Memory dynamics: `docs/core/memory_dynamics.md`
- Relation reason chain (2026-04-02): `docs/core/relation_reason_chain_2026_04_02.md`
- Relation field reconstruction summary (2026-04-02): `docs/core/relation_field_reconstruction_summary_2026_04_02.md`
- Terrain relation strategy bridge (2026-04-03): `docs/core/terrain_relation_strategy_bridge_2026_04_03.md`
- Typed timing contract (2026-04-03): `docs/core/typed_timing_contract_2026_04_03.md`
- Typed action contracts (2026-04-03): `docs/core/typed_action_contracts_2026_04_03.md`
- Typed surface context contract (2026-04-03): `docs/core/typed_surface_context_contract_2026_04_03.md`
- Typed surface runtime bridge (2026-04-04): `docs/core/typed_surface_runtime_bridge_2026_04_04.md`
- Typed runtime guidance bridge (2026-04-04): `docs/core/typed_runtime_guidance_bridge_2026_04_04.md`
- Typed integration gate export (2026-04-04): `docs/core/typed_integration_gate_export_2026_04_04.md`
- Typed runtime export boundary (2026-04-04): `docs/core/typed_runtime_export_boundary_2026_04_04.md`
- Typed expression hints contract (2026-04-04): `docs/core/typed_expression_hints_contract_2026_04_04.md`
- Typed expression hint bundles (2026-04-04): `docs/core/typed_expression_hint_bundles_2026_04_04.md`
- Typed policy packet orchestration bridge (2026-04-04): `docs/core/typed_policy_packet_orchestration_bridge_2026_04_04.md`
- AI4Animation reference bridge (2026-04-04): `docs/core/ai4animation_reference_bridge_2026_04_04.md`
- Reaction contract (2026-04-05): `docs/core/reaction_contract_2026_04_05.md`
- Subjective scene state (2026-04-07): `docs/core/subjective_scene_state_2026_04_07.md`
- Typed policy packet contract (2026-04-03): `docs/core/typed_policy_packet_contract_2026_04_03.md`
- Organism state: `docs/core/organism_state.md`
- Joint state: `docs/core/joint_state.md`
- External field state: `docs/core/external_field_state.md`
- Terrain dynamics: `docs/core/terrain_dynamics.md`
- Reaction-first response states: `docs/core/reaction_first_response_states.md`
- Recent dialogue state: `docs/core/recent_dialogue_state.md`
- Discussion thread state: `docs/core/discussion_thread_state.md`
- Issue state: `docs/core/issue_state.md`
- Discussion thread registry: `docs/core/discussion_thread_registry.md`
- Autobiographical thread: `docs/core/autobiographical_thread.md`
- LM Studio pipeline probe: `docs/core/lmstudio_pipeline_probe.md`
- Contact field + access projection model: `docs/core/contact_field_access_projection_model.md`
- Conscious workspace model: `docs/core/conscious_workspace_model.md`
- Core base spec: `docs/eqnet_core_base_spec.md`
- Observation loop + audit link: `docs/eqnet_observation_loop.md`
- Forgetting spec (reallocation, nightly-only): `docs/forgetting_spec.md`
- Forgetting runbook (ops checks): `docs/forgetting_runbook.md`
- Core validation summary: `docs/eqnet_core_validation_summary.md`
- Heart OS overview (core stack): `docs/eqnet_heart_os_overview.md`
- Inverse-problem operating model: `docs/eqnet_inverse_problem_operating_model.md`
