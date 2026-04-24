# Typed Policy Packet Orchestration Bridge (2026-04-04)

## 目的

`interaction_policy_packet` はすでに typed contract だったが、runtime と integration hooks の一部では、
そこから派生した `interaction_policy_*` の flat key を個別に読み直していた。

その結果、

- 同じ packet なのに consumer ごとに参照規則が分かれる
- current_state / gate_context / persona summary で carry の意味がずれる
- orchestration の正本が packet ではなく flat payload に戻る

という構造が残っていた。

今回の変更では、`interaction_policy_packet` を

- hint export の正本
- runtime hydration の正本
- gate carry の正本
- persona summary の正本

へ寄せた。

## 追加した helper

### `inner_os/integration_hooks.py`

- `_apply_interaction_policy_packet_views(...)`

役割:

- `interaction_policy_packet` から legacy flat key をまとめて派生
- `surface_*` の派生も同じ規則に寄せる
- `contact_reflection_state` も packet 起点に揃える

これにより、response gate 生成時の policy flat view は 1 か所で決まる。

### `emot_terrain_lab/hub/runtime.py`

- `_interaction_policy_state_map(...)`
- `_interaction_policy_state_name(...)`
- `_interaction_policy_state_bias(...)`
- `_apply_interaction_policy_packet_to_current_state(...)`
- `_apply_interaction_policy_packet_to_gate_context(...)`
- `_interaction_policy_packet_summary(...)`

役割:

- packet 下位 state の読み方を統一
- current_state への hydration を packet 起点にする
- `_last_gate_context` の carry / focus / bias を packet 起点にする
- `persona_meta["inner_os"]` の summary も packet 起点にする

## 変更の要点

### integration hooks

手で列挙していた以下を helper に移した。

- `interaction_policy_*`
- `surface_voice_texture`
- `surface_lightness_room`
- `surface_continuity_weight`
- `surface_relational_voice_texture`
- `surface_cultural_register`
- `surface_shared_moment_state`
- `surface_appraisal_state`
- `surface_listener_action_state`
- `surface_learning_mode_state`
- `surface_social_experiment_state`
- `surface_identity_arc_*`

### runtime

以下の経路で policy packet を正本に寄せた。

- `response_hook.expression_hints -> current_state`
- `response_hook.expression_hints -> _last_gate_context`
- effective interaction policy packet -> `_last_gate_context`
- effective interaction policy packet -> `persona_meta["inner_os"]`

これにより、`body_homeostasis / agenda / learning_mode / social_experiment / expressive_style / relation_competition / social_topology` の carry と summary は、同じ packet 解釈を共有する。

## テスト

- `tests/test_inner_os_integration_hooks.py`
  - `interaction_policy_contact_reflection_state`
  - `surface_voice_texture`
  - `surface_lightness_budget_state`
  が packet 由来であることを固定

- `tests/test_runtime_process_turn_hooks.py`
  - `_apply_interaction_policy_packet_to_current_state(...)`
  - `_apply_interaction_policy_packet_to_gate_context(...)`
  の direct test を追加

## 現在地

今回で、`interaction_policy_packet` は

- 生成
- legacy view 派生
- runtime hydration
- carry
- summary

の主要経路で正本に寄った。

まだ残る課題は、

- `runtime.py` 自体の責務が大きいこと
- `current_state` に残る carry field 群が多いこと
- policy packet 以外の flat legacy view をさらに orchestration 向けに整理すること

だが、少なくともこの段階で
「policy packet はあるが、実際の反応判断は flat key が握っている」
というズレはかなり薄くなった。
