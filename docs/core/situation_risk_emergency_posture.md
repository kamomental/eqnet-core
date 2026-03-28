# Situation Risk / Emergency Posture

## 目的

`scene_state` と `constraint_field` は既に強いが、これだけだと

- 同じ対象でも場面で意味が変わる
- 危険時は対話より距離化や離脱が主になる
- 信頼文脈の破断が危険意味を増幅する

を明示的に扱いにくい。

そのため `Inner OS` では、既存 core を壊さずに

1. `situation_risk_state`
2. `emergency_posture`

を sidecar として足す。

## 役割分担

### `situation_risk_state`

ここで扱うのは、対象そのものではなく

- 現在の risk token
- scene / privacy / public exposure
- routine task かどうか
- trust / familiarity / continuity
- そこから生じる逸脱感

である。

出すものは例えば

- `immediacy`
- `intent_clarity`
- `escape_room`
- `relation_break`
- `deviation_from_expected`
- `dialogue_room`

で、最終的に

- `ordinary_context`
- `guarded_context`
- `unstable_contact`
- `acute_threat`
- `emergency`

のどこに近いかを返す。

### `emergency_posture`

`situation_risk_state` を受けて、

- `observe`
- `de_escalate`
- `create_distance`
- `exit`
- `seek_help`
- `emergency_protect`

の posture に落とす。

ここで重要なのは、危険時の主判断が
「どう自然に話すか」ではなく
「対話してよいのか、距離を取るべきか、離脱を優先すべきか」
になる点である。

## 既存層との接続

- `scene_state`
  - public/private/task/場の意味を供給する
- `constraint_field`
  - boundary pressure / protective bias を供給する
- `action_posture`
  - risk posture を対話 posture へ変換する
- `actuation_plan`
  - risk posture を `create_distance / exit / seek_help` の行動列へ変換する

## 設計上の意図

これは object ontology の完成版ではない。
まずは

- 危険が object 単体で決まらない
- 同じ risk でも routine/public/private/breach で意味が変わる
- 対話と回避の優先順位を posture として分ける

ことを typed state として持つための最小層である。
