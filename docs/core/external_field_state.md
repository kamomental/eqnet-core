# External Field State

## 役割

`external_field_state` は、入力文そのものではなく、

- 関係の圧
- 場の公開性
- thread の継続力
- 曖昧さ
- 安全余地
- 新規性

をまとめた `Inner OS` の外部場 contract です。

## 含めるもの

- `dominant_field`
- `social_mode`
- `thread_mode`
- `environmental_load`
- `social_pressure`
- `continuity_pull`
- `ambiguity_load`
- `safety_envelope`
- `novelty`
- `trace`

## 設計原則

- raw observation をそのまま持たない
- `input_text` ではなく、解釈済みの場を持つ
- projection ではなく canonical state として carry できる
- `surface_context_packet` には軸だけを圧縮して渡す

## 現在の接続先

- `integration_hooks`
- `transfer_package`
- `continuity_summary`
- `runtime persona_meta / metrics`
- `surface_context_packet`

## 次の本命

- `joint_state` と接続して self-other coupling を明示化する
- `route / response_strategy / live_engagement` の upstream 判定に直接使う
