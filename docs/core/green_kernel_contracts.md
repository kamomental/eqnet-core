# Green Kernel Contracts

Date: 2026-03-28

## 目的

`Green function` を比喩で終わらせず、既存の

- memory
- affective
- relational
- boundary
- residual

を **共通内部場への射影** として扱う最小 contract を置く。

ここで Green 的に扱うのは `memory / affective / relational` の局所応答だけで、  
`boundary / residual` は別演算として残す。

## 置き方

```text
input
  -> contact
  -> event
  -> memory / affective / relational projection
  -> shared inner field
  -> readout
  -> boundary / residual
  -> surface
```

## このファイルで固定したもの

- `SharedInnerField`
  - 共通内部場への最小射影先
- `FieldDelta`
  - 各 kernel / operator の局所変形
- `GreenKernelComposition`
  - memory / affective / relational / boundary / residual を束ねた結果

## 現在の軸

- `memory_activation`
- `affective_charge`
- `relational_pull`
- `guardedness`
- `reopening_pull`
- `boundary_tension`
- `residual_tension`

これは最終理論ではなく、今の `inner_os` にある

- temporal membrane
- affect blend
- recent dialogue / discussion / issue
- boundary transform
- residual reflection
- autobiographical thread

を同じ readout 系へ寄せるための最小座標。

## 重要な制約

- 全部を 1 本の万能 Green function にしない
- `boundary / residual` は Green にせず operator として扱う
- 足し算の前に、共通内部場への射影先を固定する
- surface は shared field から一貫して生やす

## 今回の位置づけ

これは **runtime の全面置換ではなく contract の固定**。  
次段では、この composition を continuity / evaluation hook に流して、
どの kernel が何を前景化したかを比較できるようにする。
