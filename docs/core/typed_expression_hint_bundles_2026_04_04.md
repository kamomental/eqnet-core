# Typed Expression Hint Bundles (2026-04-04)

## 目的
`expression_hints` の巨大な flat payload を、そのまま内部の正本にしない。
内部では bundle contract を保持し、export 境界でだけ plain dict に落とす。

対象 bundle:

- `qualia_hint_bundle`
- `scene_hint_bundle`
- `workspace_hint_bundle`
- `interaction_reasoning_hint_bundle`
- `interaction_audit_hint_bundle`
- `field_regulation_hint_bundle`
- `terrain_insight_hint_bundle`

## 追加した contract
実装ファイル:
[expression_hint_bundles.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/expression_hint_bundles.py)

- `QualiaHintBundleContract`
- `SceneHintBundleContract`
- `WorkspaceHintBundleContract`
- `InteractionReasoningHintBundleContract`
- `InteractionAuditHintBundleContract`
- `FieldRegulationHintBundleContract`
- `TerrainInsightHintBundleContract`

各 bundle は `coerce_*` helper で contract 化し、`to_dict()` でだけ export 用の plain dict に変換する。

## 今回の整理
### `integration_hooks.response_gate(...)`
[integration_hooks.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/integration_hooks.py)

- `expression_hints` の flat key を後から再構成するのでなく、元の state / object から bundle を作る
- `ResponseGateResult.expression_hints` は内部では contract を正本にし、`to_dict()` でだけ plain dict に落とす
- scene / workspace / interaction_reasoning / interaction_audit / qualia は
  - `_apply_scene_hint_bundle_views(...)`
  - `_apply_workspace_hint_bundle_views(...)`
  - `_apply_interaction_reasoning_hint_bundle_views(...)`
  - `_apply_interaction_audit_hint_bundle_views(...)`
  - `_apply_qualia_hint_bundle_views(...)`
  で flat key と bundle を同じ規則から生成する
- `field_regulation / terrain_insight` も
  - `_apply_field_regulation_hint_bundle_views(...)`
  - `_apply_terrain_insight_hint_bundle_views(...)`
  で同じ方針に揃えた

### `runtime.py`
[runtime.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/emot_terrain_lab/hub/runtime.py)

- bundle helper の返り値を plain dict ではなく contract に変更
- `current_state` の hydration は `_apply_hint_bundles_to_current_state(...)` に統一
- `field_regulation / terrain_insight` の hydration も helper 化し、`contact/access` と `terrain/resonance` をまとめて current state に通す
- `_last_gate_context` の保持は `_apply_hint_bundles_to_gate_context(...)` に統一
- `_last_gate_context` の `contact/access` と `terrain/resonance` も bundle helper 経由に揃えた
- export 側は `_export_expression_hint_bundle_views(...)` に寄せ、bundle から legacy flat view を作る
- `persona_meta["inner_os"]` の scene / workspace / reasoning / audit 要約は bundle summary helper から生成する

## 現在の境界
- 内部保持: bundle contract
- 互換維持: flat key も残す
- export: plain dict

この 3 層を明示したことで、内部で dict に戻してからまた読む再回収を減らしている。

## まだ残る課題
- `expression_hints` 全体はまだ大きい
- legacy flat key を直接読む consumer が一部残っている
- `runtime.py` と `integration_hooks.py` 自体の責務がまだ大きい

## 評価
- `pytest tests\\test_runtime_process_turn_hooks.py tests\\test_inner_os_integration_hooks.py tests\\test_inner_os_bootstrap.py -q`
  - `154 passed, 1 warning`
- `python -m py_compile`
  - [runtime.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/emot_terrain_lab/hub/runtime.py)
  - [test_runtime_process_turn_hooks.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/tests/test_runtime_process_turn_hooks.py)

## 次の段階
- runtime の残りの flat consumer を bundle helper 起点に寄せる
- `expression_hints` の残りの flat 群を、必要に応じて sub-contract 化する
- export 境界より内側での dict 再回収をさらに削る
