# Reaction-First Response States

## 概要
発話を唯一の反応として扱わず、`Inner OS` の actuation 層で

- `speak`
- `backchannel`
- `hold`
- `defer`

を first-class の反応チャネルとして扱うための canonical state 群。

## state

### `PresenceHoldState`
- ファイル:
  - `inner_os/presence_hold_state.py`
- 役割:
  - いま場を保つべきか
  - 小さな相槌を挟めるか
  - もう少し待ってから戻れるか
- 主な入力:
  - `live_engagement_state`
  - `listener_action_state`
  - `shared_moment_state`
  - `joint_state`
  - `organism_state`
  - `external_field_state`
  - `terrain_dynamics_state`
- 主な出力:
  - `state`
  - `hold_room`
  - `reentry_room`
  - `backchannel_room`
  - `release_readiness`
  - `silence_mode`
  - `pacing_mode`

### `NonverbalResponseState`
- ファイル:
  - `inner_os/nonverbal_response_state.py`
- 役割:
  - 無発話寄りの反応を canonical に持つ
  - 将来の VAP / TTS / nod / filler 接続先を揃える
- 主な出力:
  - `response_kind`
  - `pause_mode`
  - `silence_mode`
  - `token_profile`
  - `nod_mode`
  - `breath_mode`
  - `timing_bias`

### `ResponseSelectionState`
- ファイル:
  - `inner_os/response_selection_state.py`
- 役割:
  - `primary_action` だけでなく、最終反応チャネル
    - `speak`
    - `backchannel`
    - `hold`
    - `defer`
    を選ぶ
- 主な入力:
  - `primary_action`
  - `execution_mode`
  - `reply_permission`
  - `defer_dominance`
  - `live_engagement_state`
  - `presence_hold_state`
  - `nonverbal_response_state`
  - `shared_moment_state`
- 主な出力:
  - `channel`
  - `speak_room`
  - `backchannel_room`
  - `hold_room`
  - `defer_room`

## actuation への接続
`inner_os/actuation_plan.py` で次を返すようにした。

- `presence_hold_state`
- `nonverbal_response_state`
- `response_selection_state`
- `response_channel`
- `response_channel_score`

これで `primary_action` とは別に、実際の反応チャネルを同じ packet で持てる。

## 今回の昇格
今回の更新で、reaction-first state は diagnostics だけでなく次にも通る。

- `inner_os/headless_runtime.py`
  - `response_channel`
  - `nonverbal_response_state`
  - `presence_hold_state`
- `emot_terrain_lab/hub/runtime.py`
  - `persona_meta["inner_os"]`
  - `surface_profile`
- `emot_terrain_lab/hub/llm_hub.py`
  - `[inner_os_policy].actuation_plan`
- `emot_terrain_lab/hub/lmstudio_pipeline_probe.py`
  - `actuation_response_channel`

## ねらい
- LLM を timing の本体にしない
- 無発話の反応を null 扱いしない
- state から反応チャネルを選ぶ
- 将来の MaAI / VAP / TTS 接続の正本を先に揃える

## 次
1. `runtime` で `response_channel` を実動作へ上げる
2. `hold / backchannel` を text fallback ではなく runtime action にする
3. `timing_bias / token_profile / breath_mode` を MaAI / VAP / TTS へ接続する

## 2026-04-03 relation / causal terrain を無発話反応まで拡張

- `presence_hold_state`
  - `utterance_reason_packet` を読み、`cross_context_bridge / reframing_cause / name_distant_link` などの relation reentry で `reentry_open` に入りやすくした
  - `unfinished_link / unfinished_thread_cause / keep_unfinished_link_near` などの guarded relation で `hold_room` を強めるようにした
- `nonverbal_response_state`
  - relation reentry では `bridge_ack_presence` を返し、generic な `soft_ack_presence` より先に立つようにした
  - guarded relation では `guarded_hold_presence` を返し、沈黙保持を first-class にした
- `response_selection_state`
  - `utterance_reason_packet` を直接読み、relation reentry は `backchannel`、guarded relation は `hold` に寄りやすくした
- `actuation_plan`
  - `presence_hold_state -> nonverbal_response_state -> response_selection_state` に同じ `utterance_reason_packet` を通し、
    `memory -> utterance_reason -> terrain -> hold/backchannel/speak/defer`
    が一本の chain になるようにした

### 確認

- `pytest tests\\test_inner_os_reaction_selection_state.py tests\\test_inner_os_live_engagement_state.py tests\\test_inner_os_headless_runtime.py tests\\test_runtime_process_turn_hooks.py -q`
  - `96 passed, 1 warning`

## 2026-04-03 runtime shaping まで接続

- `runtime.py`
  - `actuation_response_channel = backchannel` のときは listener token / fast ack を優先し、長い content sequence をそのまま話さないようにした
  - `actuation_response_channel = hold` のときは breath 側の minimal fallback を優先し、text-only でも沈黙寄りの reaction を保つようにした
- これで `response_channel` は
  - diagnostics
  - probe
  - headless actuation
  だけでなく final shaping にも効く

### 確認

- `pytest tests\\test_runtime_bright_short_sequence.py tests\\test_inner_os_reaction_selection_state.py tests\\test_inner_os_headless_runtime.py tests\\test_runtime_process_turn_hooks.py -q`
  - `83 passed, 1 warning`

## 2026-04-02 clarify opening sticky
- `force_llm_bridge` の clarify 系では、`offer_small_opening_line / offer_small_opening_frame` を含む sequence を `opening_support` shape として保持するようにした
- これにより bright 系の `bright_bounce` が後段で上書きせず、
  - `切り出すなら…`
  - `まだうまく整理できないなら…`
  のような small opening support を final まで保てる
- locale cue も拡張し、`どこから話せば / 何から話せば` 系の opening request を `interaction_policy_packet.opening_request_hint` で拾えるようにした
