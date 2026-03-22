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
- Contact field + access projection model: `docs/core/contact_field_access_projection_model.md`
- Conscious workspace model: `docs/core/conscious_workspace_model.md`
- Core base spec: `docs/eqnet_core_base_spec.md`
- Observation loop + audit link: `docs/eqnet_observation_loop.md`
- Forgetting spec (reallocation, nightly-only): `docs/forgetting_spec.md`
- Forgetting runbook (ops checks): `docs/forgetting_runbook.md`
- Core validation summary: `docs/eqnet_core_validation_summary.md`
- Heart OS overview (core stack): `docs/eqnet_heart_os_overview.md`
- Inverse-problem operating model: `docs/eqnet_inverse_problem_operating_model.md`
