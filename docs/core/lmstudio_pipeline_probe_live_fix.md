# LM Studio Probe Live Fix

LM Studio live 経路で、`content_sequence` は正しいのに最終文面が raw bridge 側へ戻る問題に対して、
以下の確認項目を probe で直接見えるようにした。

- `planned_content_sequence_present`
  - 最終 shaping 時点で planned sequence を保持していたか
- `allow_guarded_narrative_bridge`
  - raw LLM 文面を補助的に混ぜる許可が最終判断で残っているか
- `guarded_narrative_bridge_used`
  - 実際に raw bridge が最終文面へ混入したか

今回の修正では、

1. live response loop の最後で persistent な planning hint を再適用する
2. `offer_small_opening_line` がある場合は guarded narrative bridge を落とす
3. planned sequence が落ちても、元のユーザー入力から deterministic に再構成する

ようにした。

これにより、`どう切り出せば` 系の要求では raw 助言文より
`切り出すなら…` の opening line が最終文面の正本になる。
