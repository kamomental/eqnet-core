# Issue State

`issue_state` は、`recent_dialogue_state` と `discussion_thread_state` の上に置く、
さらに短い論点位相の sidecar です。

主眼は、

- いま論点を探っているのか
- いったん置いておく流れなのか
- 収まりつつあるのか

を、prompt 上の自然文解釈だけに任せず typed state として持つことです。

## 役割

`issue_state` は次を分離します。

- `naming_issue`
  - 論点をまだ名指している段階
- `exploring_issue`
  - 未解決点を少し掘っている段階
- `pausing_issue`
  - いったん置いて return point を残す段階
- `resolving_issue`
  - 収まりつつあり、閉じすぎず着地へ向かう段階

## 入力

- 現在の user text
- 直近 history
- `discussion_thread_state`
- `recent_dialogue_state`
- `interaction_policy`

## 出力

- `state`
- `issue_anchor`
- `question_pressure`
- `pause_readiness`
- `resolution_readiness`
- `dominant_inputs`

## 使い方

現状では sidecar として planner / turn delta / continuity summary に流しています。

- `pausing_issue`
  - `leave_return_point` や `protect_talking_room` を押しやすくする
- `exploring_issue`
  - `stay_with_present_need` を押しやすくする

まだ議論 registry ではありません。
まずは same-turn の論点位相を持つための最小層です。
