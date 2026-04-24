# Epistemic State

`epistemic_state` は、知識を大量に持つ層そのものではなく、
**その知識を今どれだけ信用してよいか** を表す小さい状態契約です。

## 目的

knowledge cutoff を「巨大な静的知識で埋める」のでなく、

- いまの知識は新しいか
- どの出典にどれだけ寄っているか
- 変化しやすい対象か
- いま再検証を優先すべきか

を slow-state ではなく **知識運用状態** として切り出します。

## 中心軸

- `freshness`
  - その知識がどれだけ新しいか
- `source_confidence`
  - 出典や取得経路をどれだけ信用しているか
- `verification_pressure`
  - いま再検証を優先すべき圧
- `change_likelihood`
  - 対象が変わりやすいか
- `stale_risk`
  - 古くなっている危険
- `epistemic_caution`
  - 断定を避けるべき度合い

## posture

- `carry_forward`
  - そのまま使ってよい
- `reverify`
  - 再検証を先に置く
- `hold_ambiguity`
  - 断定せず保留する
- `update_priority`
  - 更新を優先する

## 境界

- `inner_os`
  - `epistemic_state` の contract と更新則
- `memory / rag / web / vlm`
  - 知識の取得元
- `llm_hub / surface_context_packet`
  - 断定の強さや再検証方針を表出へ運ぶ接合部

つまり `epistemic_state` は知識そのものではなく、
**知識の鮮度と扱い方の state** です。
