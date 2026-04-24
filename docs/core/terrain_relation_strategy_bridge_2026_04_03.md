# Terrain Relation Strategy Bridge (2026-04-03)

## 要点

`memory_dynamics` の relation / causal field を `terrain_dynamics` に通しただけでは、
まだ上流の反応選択は変わり切らない。

今回の変更では、`terrain_dynamics` から得られる

- `dominant_basin`
- `dominant_flow`
- `barrier_height`
- `entropy`
- `recovery_gradient`
- `basin_pull`

を、`response_strategy` と `habit` shortcut の抑制に直接使うようにした。

これで chain は次の形になる。

`memory_dynamics -> appraisal -> meaning_update -> utterance_reason -> terrain_dynamics -> strategy / route`

## 追加した橋渡し

### 1. policy_packet の field strategy override

`_derive_field_strategy_override(...)` に、`utterance_reason_packet` の

- `relation_frame`
- `causal_frame`
- `memory_frame`
- `preserve`

を読み取る経路を追加した。

そのうえで、terrain の状態と組み合わせて 2 つの override を導入した。

- `field_relation_guard`
  - `unfinished_link`
  - `unfinished_thread_cause`
  - `keep_unfinished_link_near`
  のような guarded relation が立っていて、
  terrain が `steady/protective/diffuse` 側に傾いているときは
  `respectful_wait` に落とす。
- `field_relation_reentry_progression`
  - `cross_context_bridge`
  - `reframing_cause`
  - `name_distant_link`
  のような relation reentry が立っていて、
  terrain が `continuity/recovery` 側に開いているときは
  `shared_world_next_step` に上げる。

つまり、shared moment が強いときだけでなく、
relation / causal reentry 自体が strategy を動かせるようになった。

### 2. runtime の habit suppression

`EmotionalHubRuntime._should_use_habit(...)` に、
`_has_recent_relation_field_bias()` を追加した。

ここでは直近 packet の

- `utterance_reason_packet`
- `terrain_dynamics_state`

を読んで、

- guarded relation が protective terrain に残っている
- relation reentry が continuity / recovery terrain に開いている

場合は `habit` shortcut を使わず、conscious 側へ残す。

これで bright continuity だけでなく、
memory / causal reentry でも `route` が安易に `habit` へ落ちにくくなった。

## 評価

追加した回帰:

- `tests/test_inner_os_live_engagement_state.py`
  - relation reentry が `shared_world_next_step` を引くこと
  - guarded relation が `respectful_wait` を引くこと
- `tests/test_runtime_route_prompts.py`
  - relation / causal terrain bias があると `habit` route を suppress すること

確認:

- `pytest tests\\test_inner_os_live_engagement_state.py tests\\test_runtime_route_prompts.py -q`
  - `26 passed, 1 warning`
- `pytest tests\\test_runtime_process_turn_hooks.py tests\\test_inner_os_integration_hooks.py -q`
  - `128 passed, 1 warning`

warning は既存の `python_multipart` のみ。

## 現在地

ここまでで、

- `memory_dynamics`
- `utterance_reason`
- `terrain_dynamics`
- `response_strategy`
- `route`

のあいだに、最小の relation-aware / causal-aware な一本線が通った。

## 2026-04-03 posture / actuation まで拡張

この round では、同じ relation / causal terrain bias を

- `action_posture`
- `actuation_plan`

にも通した。

### 追加したもの

- relation reentry progression
  - `cross_context_bridge`
  - `reframing_cause`
  - `name_distant_link`
  のような signal が continuity / recovery terrain 上で立つと、
  `shared_progression` に上げる
- guarded relation hold
  - `unfinished_link`
  - `unfinished_thread_cause`
  - `keep_unfinished_link_near`
  のような signal が protective / diffuse terrain 上で立つと、
  `defer_with_presence` に落とす

これで

`memory -> reason -> terrain -> strategy -> posture -> actuation`

までが、同じ正本でつながった。

### 追加した回帰

- `tests/test_inner_os_live_engagement_state.py`
  - shared moment がなくても relation reentry だけで `co_move / shared_progression` に入ること
  - social pressure field がなくても guarded relation だけで `wait / defer_with_presence` に入ること

確認:

- `pytest tests\\test_inner_os_live_engagement_state.py -q`
  - `19 passed, 1 warning`
- `pytest tests\\test_runtime_process_turn_hooks.py tests\\test_inner_os_headless_runtime.py tests\\test_runtime_route_prompts.py -q`
  - `78 passed, 1 warning`

## 残る本命

- `hold / backchannel / nonverbal response` を relation-conditioned にすること
- MaAI/VAP などの timing 層へ `response_channel / wait_before_action` をつなぐこと
- live probe で raw / final / route / actuation の整合を確認すること

## 2026-04-03 reaction-first channel まで接続

- relation / causal terrain の影響を `strategy / route` で止めず、
  - `presence_hold_state`
  - `nonverbal_response_state`
  - `response_selection_state`
  まで下ろした
- これにより、
  - relation reentry は `reentry_open -> bridge_ack_presence -> backchannel`
  - guarded relation は `holding_space -> guarded_hold_presence -> hold`
  のように、shared moment がなくても current relation field から無発話反応が立つ
- 実装上は `actuation_plan` が同じ `utterance_reason_packet` を 3 state に通す形にそろえた

### 回帰

- `pytest tests\\test_inner_os_reaction_selection_state.py tests\\test_inner_os_live_engagement_state.py tests\\test_inner_os_headless_runtime.py tests\\test_runtime_process_turn_hooks.py -q`
  - `96 passed, 1 warning`
