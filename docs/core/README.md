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

- EQ core v2 operating concept: `docs/core/eq_core_v2_operating_concept.md`
- EQ core human ergonomics framework: `docs/core/eq_core_human_ergonomics_framework.md`
- EQ core reconstruction for risk reduction: `docs/core/eq_core_reconstruction_for_risk_reduction.md`
- Conversational object architecture: `docs/core/conversational_object_architecture.md`
- Interaction option search architecture: `docs/core/interaction_option_search_architecture.md`
- Qualia contact-access model: `docs/core/qualia_contact_access_model.md`
- Qualia membrane projection model: `docs/core/qualia_membrane_projection_model.md`
- Qualia structure + temporal memory integration: `docs/core/qualia_structure_temporal_integration.md`
- EQNet boundary / residual layers: `docs/core/eqnet_boundary_residual_layers.md`
- Emergency expression bridge: `docs/core/emergency_expression_bridge.md`
- Deep disclosure reflection: `docs/core/deep_disclosure_reflection.md`
- Contact reflection state: `docs/core/contact_reflection_state.md`
- Green kernel contracts: `docs/core/green_kernel_contracts.md`
- Surface expression selection: `docs/core/surface_expression_selection.md`
- Surface context packet: `docs/core/surface_context_packet.md`
- Codex use-case environment: `docs/core/codex_usecase_environment.md`
- Codex plugin root: `docs/core/codex_plugin_root.md`
- Route surface priority: `docs/core/route_surface_priority.md`
- Evaluation criteria: `docs/core/evaluation_criteria.md`
- Evaluation snapshot (2026-03-28): `docs/core/evaluation_snapshot_2026_03_28.md`
- Evaluation targets: `docs/core/evaluation_targets.md`
- Evaluation loop (anchor progress, 2026-03-28): `docs/core/evaluation_loop_anchor_progress_2026_03_28.md`
- Mechanism issue map: `docs/core/mechanism_issue_map.md`
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
