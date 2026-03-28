# EQNet Boundary / Residual Layers

このメモは、`eqnet core` の内的共鳴系を直接書き換えずに、
既存の候補生成・reportability・constraint の上に
`boundary transform` と `residual reflection` を薄く重ねるための整理です。

## 位置づけ

- 内的共鳴系
  - 感情地形
  - 関係履歴
  - 記憶場
  - 自己連続性
- 境界変換層
  - 候補をそのまま出すのでなく
  - 通す / 弱める / 保留する / 閉じる
  - を接触条件として扱う
- 残差反映層
  - 出せなかった候補
  - 弱められた候補
  - 閉じたまま残した話題
  - を次の内部更新へ返す入口にする

## 今回の実装方針

今回は既存挙動を置き換えない。

- `boundary_transformer`
  - `content_sequence`
  - `interaction_constraints`
  - `conversation_contract`
  - `constraint_field`
  - `reportability_gate`
  から、候補が境界でどう扱われるかを typed sidecar として整理する
- `residual_reflector`
  - `boundary_transform` の結果から
  - withheld / softened / deferred を残差としてまとめる

## 重要な不変条件

1. 内部状態は出力に還元しない
2. 規範は core ではなく境界に作用させる
3. 抑制や保留の差分を残差として残す
4. 既存の `constraint_field` / `reportability_gate` を壊さない

## 現段階の限界

- 今回の `boundary_transform` は、まず観測と整流の sidecar であり、
  表面挙動の全面置換ではない
- `authority resolver` はまだ独立していない
- `residual reflection` は carry への入口であり、
  まだ長期更新の正本にはしていない

## 次段の接続先

- `distillation_record`
- `continuity_summary`
- `daily_carry_summary`
- runtime `controls_used`

ここへ順に流すことで、
「何が出たか」だけでなく
「何が出ずに残ったか」を運用上も読めるようにする。
