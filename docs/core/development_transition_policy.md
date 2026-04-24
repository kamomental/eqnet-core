# Development Transition Policy

`development_transition_policy` は、
既存の slow-state 断片を `growth_state` に束ねる更新則です。

## 入力

- `development_state`
- `forgetting_snapshot`
- `sleep_consolidation`
- `transfer_package`

## 役割

これまでは

- 成長
- 忘却
- 再固定化
- carry

が別々の断片として存在していました。

この policy は、それらを 1 本の slow-state 遷移として読みます。

## 方針

- 新しい巨大モジュールは作らない
- 既存 core の上に薄く乗せる
- hard threshold ではなく blend を使う
- `social_update_strength` と `identity_update_strength` を
  更新強度としてそのまま利用する

## いま束ねているもの

- 関係の更新
  - `belonging`, `trust_bias`
- 知識成熟
  - `abstraction_readiness`, `reconsolidation_priority`
- 表出 carry
  - `expressive_style_carry_bias`, `lexical_variation_bias`
- lingering の統合
  - `replay_priority`, `autobiographical_pull`, `monument_salience`
- 一貫性
  - `identity_preservation_bias`, `continuity_score`

## まだ束ねていないもの

- `epistemic_state`
  - freshness / verification / source を持つ知識更新層
- `discourse_shape`
  - 表出の談話骨格

つまりこれは、アマデウス級への入口であって、
統合の最終形ではありません。
