# Typed Action Contracts (2026-04-03)

## 目的

`action_posture` と `actuation_plan` は、状態更新系の下流にある重要な境界だが、
実装上は巨大な `dict[str, Any]` を返しており、正本が packet 側へ再回収されやすかった。

今回はこの問題に対して、

- `ActionPostureContract`
- `ActuationPlanContract`

を追加し、**返り値の正本だけを typed dataclass に寄せる** 方針を取った。

## 変更内容

### 1. Mapping 互換の dataclass を追加

以下の返り値を `Mapping` 互換の dataclass にした。

- `inner_os/action_posture.py`
  - `ActionPostureContract`
  - `coerce_action_posture_contract(...)`
- `inner_os/actuation_plan.py`
  - `ActuationPlanContract`
  - `coerce_actuation_plan_contract(...)`

`[...]`、`.get(...)`、`dict(...)` を壊さないため、contract 自体は `Mapping[str, object]` として振る舞う。

### 2. core field と extras を分離

すべてを一気に typed 化せず、まずは境界として重要なキーだけを core field にした。

#### ActionPostureContract の core 例

- `engagement_mode`
- `outcome_goal`
- `boundary_mode`
- `next_action_candidates`
- `attention_target`
- `memory_write_priority`
- `ordered_operation_kinds`
- `ordered_effect_kinds`
- `question_budget`
- `utterance_reason_relation_frame`
- `joint_mode`
- `terrain_basin_name`

#### ActuationPlanContract の core 例

- `execution_mode`
- `primary_action`
- `action_queue`
- `reply_permission`
- `wait_before_action`
- `repair_window_commitment`
- `response_channel`
- `response_channel_score`
- `presence_hold_state`
- `nonverbal_response_state`
- `response_selection_state`

それ以外の広い payload は `extras` に残している。

## 設計意図

今回やりたかったのは、`if` を増やすことではない。

やりたかったのは、

- 判断を `dict` の寄せ集めから切り離す
- 返り値の意味境界を先に固定する
- 既存 runtime / test / probe の互換を壊さない

の3点である。

そのため、外向きにはまだ `dict(...)` へ落とせる形を保ちつつ、
**内部の正本だけを contract に寄せる** 中間段階にした。

## まだ残っている課題

これは第一段であり、以下はまだ残っている。

- `interaction_policy` 自体は依然として巨大 packet
- `surface_context_packet` も広い `dict` 依存が残る
- `runtime.py` と `integration_hooks.py` の God object 化は未解消
- `bootstrap` 系テストは payload 依存がまだ強い

つまり、今回は
**state-first を崩している巨大 `dict` のうち、action 系の返り値境界だけを先に締めた**
段階である。

## 確認

- `py_compile`
  - `inner_os/action_posture.py`
  - `inner_os/actuation_plan.py`
  - `inner_os/__init__.py`
  - `inner_os/expression/models.py`
  - `tests/test_inner_os_live_engagement_state.py`
- `pytest tests/test_inner_os_live_engagement_state.py tests/test_inner_os_reaction_selection_state.py tests/test_inner_os_conversational_architecture.py tests/test_inner_os_bootstrap.py tests/test_inner_os_integration_hooks.py -q`
  - `132 passed, 1 warning`

warning は既存の `python_multipart` のみ。
