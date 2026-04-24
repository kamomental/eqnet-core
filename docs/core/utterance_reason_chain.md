# Utterance Reason Chain

`utterance_reason_chain` は、

- 何が起きたか
- それをどう評価したか
- その結果なにが少し変わったか
- だから今回は何を言うか

を、1 本の遷移として扱うための最小層です。

## 背景

これまでの bright / live では、

- `shared_moment`
- `listener_action`
- `live_engagement`

までは持てていました。  
ただし、

- なぜ `ふふっ` なのか
- なぜ質問せず短く受けるのか
- なぜ解釈や一般論に広げないのか

の理由は、canonical な state としては弱いままでした。

## 追加した 3 層

### 1. `appraisal_state`

現在の cue と直前までの流れを照合して、

- どんな背景が続いていたか
- 今回の小さい出来事は何か
- 何が少しほどけたか

を表します。

例:

- `background_state = awkwardness_present`
- `moment_event = laugh_break`
- `shared_shift = shared_smile_window`

### 2. `meaning_update_state`

`appraisal_state` を受けて、

- self
- relation
- shared world

のどこが少し変わったかを持ちます。

例:

- `self_update = guard_relaxes_for_moment`
- `relation_update = shared_smile_window`
- `world_update = topic_not_only_strain`

### 3. `utterance_reason_packet`

実際に何を差し出すかを決める層です。

例:

- `reaction_target = small_laugh_moment`
- `offer = brief_shared_smile`
- `preserve = keep_it_small`

## 位置づけ

理想の流れは次です。

`cue + memory + relation + body`
-> `appraisal_state`
-> `meaning_update_state`
-> `utterance_reason_packet`
-> `act`
-> `discourse_shape`
-> `surface_realization`

## 現在の接続

今は以下まで通っています。

- `policy_packet`
- `surface_context_packet`
- `runtime`
- `LLMHub` の `[inner_os_policy]` prompt

つまり raw / final の両方が、少なくとも同じ「理由鎖」を参照できる入口ができています。

## 次の段階

まだ未完なのは、

- `appraisal_state` を palace / monument / ignition と直接つなぐこと
- `joint_state` に入れて common ground の更新として扱うこと
- `discourse_shape` が `utterance_reason_packet` を主因として選ばれること

です。
