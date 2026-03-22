# Inner OS Learning Contract

## Core

この系で学習させる正本は、言語モデルの内部状態ではない。

正本は、外に出して読める意味状態である。

- `h`
  - 何を観測したか
- `I`
  - 観測からどう自己を推定したか
- `Π_q`
  - その場でどの感じが立ったか
- `z_aff`
  - 感情地形のどこにいるか
- `terrain_readout`
  - その場所がどういう場所か
- `protection_mode`
  - どう守るか
- `commitment_state`
  - その場でどちらへ向きが決まったか
- `memory_write_class`
  - 何として残すか
- `daily_carry_summary`
  - 翌日に何が残るか

## Invariant

- hidden state は計算用であり、学習の正本にしない
- 学習の中心は next-token ではなく next-state / next-decision / next-carry である
- same-turn の判断と post-turn の更新と overnight の carry を混ぜない
- 言語モデルは最後の表面化を担当し、意味状態の決定を担当しない
- distillation には raw latent ではなく structured record を使う

## Learning Split

### 1. Same-turn

学習対象:

- `I`
- `Π_q`
- `z_aff`
- `terrain_readout`
- `workspace`
- `protection_mode`
- `commitment_state`

主な目的:

- 次の判断を当てる
- 判断の margin を当てる
- safety 条件を破らない

### 2. Post-turn

学習対象:

- `terrain_plasticity`
- `association` reinforcement
- `initiative_followup_bias`

主な目的:

- 何が次ターンへ残るか
- どの局所地形が変わるか
- どの link が強まるか

### 3. Overnight

学習対象:

- `daily_carry_summary`
- `commitment` carry
- `insight` carry
- `temperament` carry

主な目的:

- 翌日に何が弱い prior として戻るか

## Distillation Rule

distillation の一次正本は `inner_os_distillation_record` とする。

この record は、少なくとも次を持つ。

- どの model で答えたか
- `model_source`
- 入力 fingerprint
- same-turn の decision snapshot
- overnight との carry snapshot
- 出力 fingerprint

平文テキストは既定では残さない。

必要なら opt-in で別経路にする。

## Declaration

このプロジェクトでは、

- 状態は LM の内部に閉じ込めず
- 外部の意味状態として保持し
- その更新則を structured log から学習し
- 最終文だけを小さい言語モデルで表面化する

この流れを正本として実装と学習を進める。
