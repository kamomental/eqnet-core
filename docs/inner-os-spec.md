# Inner OS Spec v0.1

## 1. Purpose
`Inner OS` は、共生型感情知能のための構造化内部状態層である。  
責務は次の 5 つに限定する。

- grounding
- continuity
- value regulation
- access selection
- expression handoff

LLM は `Inner OS` そのものではない。  
LLM は、前景化された状態を人間向けに表出する downstream layer である。

## 2. Core design principles
1. Grounded before verbal
2. State before style
3. Access before expression
4. Continuity before personalization
5. Replaceable hypotheses, stable runtime core

## 3. Main layers

### 3.1 Grounding Layer
Responsibilities:

- 観測を grounded entities へ変換する
- affordance を付与する
- symbol を percept / action / value に接地する

Inputs:

- camera
- audio
- sensor streams
- context tokens

Outputs:

- `ObservationBundle`
- `AffordanceMap`
- `SymbolGroundingMap`
- observation uncertainty

### 3.2 State Core Layer
Responsibilities:

- world state を保つ
- self state を保つ
- continuity state を保つ

World state examples:

- scene graph
- spatial map
- object states
- task states
- relation edges

Self state examples:

- arousal
- fatigue
- trust
- curiosity
- task load
- safety margin

Continuity state examples:

- stable traits
- adaptive traits
- continuity history
- ambiguity

### 3.3 Value & Access Layer
Responsibilities:

- emotion terrain / value gradient を計算する
- attention を更新する
- reportable foreground を選ぶ

Outputs:

- `ValueState`
- `ForegroundState`
- report candidates
- memory fixation candidates

### 3.4 Expression Bridge
Responsibilities:

- `ForegroundState` を発話・multimodal response に変換する

Constraints:

- raw observation を LLM に直接渡さない
- LLM は `ForegroundState + dialogue_context` のみを見る

## 4. Identity continuity requirement
このシステムは one-shot classifier ではなく、継続する person を扱う。

Person identity は次で表現する。

- stable traits
- adaptive traits
- continuity history
- uncertainty

evidence が曖昧なとき、forced identity assignment はしない。

## 5. Minimal equations

```text
dx/dt = -∇Φ(x, t) + B u(t) - Γ L_G x + ξ(t)
Z_t = Select(W_t, X_t, Φ_t, Attention_t)
utterance = LLM(Z_t, dialogue_context)
```

Where:

- `x`: internal state
- `W_t`: world state
- `X_t`: self / continuity state
- `Φ_t`: value terrain
- `Z_t`: accessible foreground

## 6. Subjective access stance
v0.1 では、主観アクセスを reportable / controllable / memory-relevant な前景選択として扱う。  
クオリア膜や低次元界面は有望な仮説だが、コア runtime をそれに依存させない。

## 7. Non-goals in v0.1
- 完成したクオリア理論
- 2 次元膜の必然性の証明
- AGI の最終アーキテクチャ
- 生体情報の平文長期保存
