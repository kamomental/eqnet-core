# Inner OS 再構成設計 v0.1

## 目的
この文書は、既存 `inner_os` と `EQNet Core Emotional DFT` の議論を、実装境界が崩れない形で再構成するための設計書である。

狙いは 2 つある。

1. LLM を知能本体にしない
2. アフォーダンス、記号接地、意識アクセスを状態更新系として切り出す

## 設計原則
1. 世界・自己・価値・表出を分ける
2. 記号は知覚・価値・行為へ戻れるようにする
3. LLM は前景状態だけを見る
4. 感情はスタイルではなく制御変数として扱う
5. 主観仮説は交換可能なモジュールに留める

## 4 層圧縮
`Inner OS` は次の 4 層へ圧縮して扱う。

### 1. Grounding
責務:

- 観測の統合
- アフォーダンス推定
- 記号接地

入出力:

- input: sensor streams, tokens, context
- output: `ObservationBundle`, `AffordanceMap`, `SymbolGroundingMap`

### 2. State Core
責務:

- world state
- self state
- continuity state
- uncertainty

`world state` の例:

- scene graph
- spatial map
- object state
- task state
- relation graph

`self state` の例:

- arousal
- fatigue
- trust
- curiosity
- task load
- safety margin

`continuity state` の例:

- person registry
- stable traits
- adaptive traits
- ambiguity

### 3. Value & Access
責務:

- emotion terrain
- attention
- access selection
- reportable foreground

ここでの要点:

- 感情は `Φ(x)` として価値地形に置く
- access は哲学的意識そのものではなく、reportable / controllable / memory-relevant な前景選択として定義する

### 4. Expression Bridge
責務:

- `ForegroundState` を発話や multimodal response に変換する

制約:

- raw observation を LLM に直接渡さない
- LLM は `ForegroundState + dialogue_context` だけを見る

## アフォーダンス
`Grounding` の本体は物体ラベルではなく行為可能性である。

例:

```text
cup:
  can_grasp
  can_drink
  can_handover
  spill_risk
```

つまり世界モデルは「名詞の集合」ではなく「行為可能世界」でなければならない。

## 記号接地
語は文字列のままではなく、知覚・価値・行為の束として持つ。

```text
symbol =
  lexical_form
  perceptual anchors
  affordance links
  value links
  action policies
  social meaning
```

これにより `"危ない"` が

- 高速接近
- safety margin の低下
- 回避優先
- 警告発話

へ戻れる。

## 意識アクセス
ここでは意識アクセスを次で定義する。

```text
Z_t = Select(W_t, X_t, Φ_t, Attention_t)
```

- `W_t`: world state
- `X_t`: self / continuity state
- `Φ_t`: value terrain
- `Z_t`: foreground state

これは主観経験の最終理論ではないが、工学的には十分に強い定義である。

## Emotional DFT の位置
`Emotional DFT` は `Value & Access` 層を厚くする理論であり、`Expression Bridge` を肥大化させるものではない。

分離:

- Green function: 刺激応答
- terrain: 価値勾配
- access interface: 前景選択の幾何仮説
- DFT: 動的構造の解析層

## Continuity
長期共生のため、`State Core` には one-shot classifier ではなく continuity layer が必要である。

人は

- 髪型が変わる
- 老化する
- 発話が弱くなる
- 病気で歩容が変わる

ので、`Person_ID` は固定テンプレートではなく、変化を含んだ連続性として管理する。

```text
Person_ID = identity core + adaptive shell + continuity history
```

## Core / Replaceable / Experimental

### Core
- 状態方程式
- world/self/continuity separation
- value terrain
- foreground selection
- telemetry / eval

### Replaceable
- subjective model
- access geometry hypothesis
- terrain parameterization
- DFT / Koopman / graph spectral

### Experimental
- 2 次元固定の必然性
- クオリア膜の強い主張
- 解離を特異点で読む強い解釈

## まとめ
`Inner OS` の再構成で最も重要なのは、LLM を会話の上流に置かないことだ。  
本体は

- 接地された世界モデル
- 内的状態
- 価値地形
- 前景選択

であり、LLM はそれを人に伝わる形へ変換する最終段に留める。
