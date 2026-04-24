# Listener Action State

`listener_action_state` は、`うん / うんうん / そうだね / ふふっ` のような
相槌・軽い笑い・小さな受けを、後段の文面装飾ではなく
`Inner OS` の canonical state として扱うための層です。

## 役割

- `shared_moment_state`
  - ちょっと笑えた
  - 少し楽になった
  - ちいさなうれしさがあった
- `live_engagement_state`
  - pick up するか
  - riff するか
  - hold するか
- `expressive_style_state`
  - playful か
  - warm か
  - careful か
- `cultural_conversation_state`
  - casual/shared か
  - careful/polite か

をまとめて、**今このターンでどの listener action family が自然か**
を決めます。

## 出力

- `state`
  - `none`
  - `soft_ack`
  - `playful_ack`
  - `warm_laugh_ack`
- `filler_mode`
  - `professional`
  - `caregiver`
  - `playful`
- `token_profile`
  - `plain_ack`
  - `soft_ack`
  - `double_ack`
  - `soft_laugh`
- `acknowledgement_room`
- `laughter_room`
- `filler_room`

## 設計意図

ここで重要なのは、**トークンそのものを決めない**ことです。
`listener_action_state` は

- 何を感じたか
- どのくらい joint か
- どれくらい playful に行けるか

を canonical に持つだけで、実際の

- `うん`
- `うんうん`
- `そうだね`
- `ふふっ`

などの表現は、後段の filler/backchannel 系や surface realization に渡します。

## 位置づけ

- `shared_moment_state`
  - 小さな共有出来事の state
- `listener_action_state`
  - その出来事にどう受けるかの state
- `surface_context_packet`
  - LLM / renderer に渡す packet

という順でつながります。
