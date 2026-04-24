# Growth State

`growth_state` は、既存の slow-state 断片を 1 本の上位状態として読むための
小さな canonical contract です。

## 目的

この repo にはすでに以下の断片があります。

- `development_core`
- `forgetting_core`
- `sleep_consolidation_core`
- `transfer_package`

不足していたのは、それらをまとめて
「この存在がどう育っているか」を読む上位状態でした。

`growth_state` はそのための薄い境界です。

## 軸

- `relational_trust`
  - 関係の持続と再開のしやすさ
- `epistemic_maturity`
  - 不確実性と再固定化を扱う成熟度
- `expressive_range`
  - 表出の幅
- `residue_integration`
  - 未表出や lingering を統合できている度合い
- `playfulness_range`
  - 明るさや軽い共鳴へ移れる幅
- `self_coherence`
  - 内的一貫性

## 位置づけ

- `inner_os`
  - reusable な slow-state contract
- `emot_terrain_lab`
  - runtime/live での更新・観測・表出
- `emotional_dft / eqnet_core`
  - 場・qualia・terrain 仮説

つまり `growth_state` は、
全部を `inner_os` に移すためではなく、
既存断片を reusable core として読むための境界です。

## replay との関係

`docs/replay` 側にはすでに

- `bond`
- `stability`
- `curiosity`

の growth 表示があります。

`growth_state` はそれを壊さず、
`to_replay_axes()` で表示系へ射影できるようにしています。

## 次

次の issue は `AMA-03` で、
`growth_state` をどう更新するかを `development_transition_policy` に固定します。
