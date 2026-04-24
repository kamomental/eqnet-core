# Heartbeat Structure State

`heartbeat_structure_state` は、単なる `heart rate / phase` の表示ではなく、
内部状態と表出反応のあいだに置く canonical な心拍反応状態です。

## 目的

画面上の heartbeat を

- 瞬間値のメーター
- 固定の pulse アニメーション
- 外部から与えた離散ラベル

として扱うのではなく、

- いまどれだけ反応が立ち上がっているか
- いまは寄り添うべきか、保護すべきか、少し弾めるか
- 直前からどのように揺れたか

を持つ時系列 state にするための層です。

## 入力

- `heart_snapshot`
  - `rate`
  - `phase`
- `metrics`
  - `life_indicator`
  - `tension_score`
  - `phi_norm`
  - `heart_rate_norm`
- `qualia_structure_state`
- `qualia_planner_view`
- `growth_state`
- `previous_state`

## 出力

- `pulse_band`
  - `soft_pulse / lifted_pulse / racing_pulse`
- `phase_window`
  - `downbeat / upswing / crest / release`
- `activation_drive`
- `attunement`
- `containment_bias`
- `recovery_pull`
- `bounce_room`
- `response_tempo`
- `entrainment`
- `dominant_reaction`
  - `steady / attune / contain / recover / bounce`
- `trace`

## packet 軸

`to_packet_axes()` は表出層へ次の軸を渡します。

- `activation`
- `attunement`
- `containment`
- `recovery`
- `tempo`

これにより heartbeat は、見た目だけの鼓動ではなく、
`surface_context_packet` や continuity summary に接続される
内部状態になります。

## 現在の位置づけ

この state はすでに

- runtime guidance
- `surface_context_packet`
- runtime metrics
- transfer package
- continuity summary

に配線されています。

まだ explorer/UI がこれを主表示にしていない場合でも、
backend では「反応としての heartbeat」を扱える土台になっています。
