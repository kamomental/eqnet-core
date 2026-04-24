# AI4Animation Reference Bridge (2026-04-04)

## 目的

AI4Animation の原理を、そのまま motion synthesis として持ち込むのではなく、
`Inner OS` の

- 状態更新
- 反応選択
- timing
- scene interaction

の設計原理へ翻訳する。

この文書の前提は、

- LLM を本体にしない
- state / dynamics を正本にする
- 発話は最後に落とす

である。

## 参照した一次資料

- AI4Animation README
  - [GitHub](https://github.com/sebastianstarke/AI4Animation/blob/master/README.md)
- 主要研究トラック
  - Phase-Functioned Neural Networks for Character Control (SIGGRAPH 2017)
  - Mode-Adaptive Neural Networks for Quadruped Motion Control (SIGGRAPH 2018)
  - Neural State Machine for Character-Scene Interactions (SIGGRAPH Asia 2019)
  - Local Motion Phases for Learning Multi-Contact Character Movements (SIGGRAPH 2020)
  - DeepPhase: Periodic Autoencoders for Learning Motion Phase Manifolds (SIGGRAPH 2022)
  - Categorical Codebook Matching for Embodied Character Controllers (SIGGRAPH 2024)

## こちらに効く原理

### 1. phase は mode より細かい連続制御変数

AI4Animation の流れでは、phase は単なるラベルではなく、
時系列のどこにいるかを示す連続的な制御変数として使われる。

こちらに写すと、

- `bright`
- `repair`
- `hold`

のような離散 mode を先に選ぶのでなく、

- 入り始め
- 保持中
- 解放中
- 再接近中

のような局所相を連続量で持つ方が自然である。

### 2. if の森より、state 条件付きの重み付け

Mode-Adaptive Neural Networks は、
固定重みで 1 本の if 木を回すのでなく、
現在状態に応じて重みを動的に混ぜる。

こちらでは、

- `response_selection_state`
- `action_posture`
- `actuation_plan`

を、rule の貼り増しでなく
`organism / joint / terrain / timing` 条件付きの expert blending に寄せる示唆になる。

### 3. scene interaction は scene bundle と controller の往復

Neural State Machine は、scene geometry と control を切り離さず、
scene interaction を single controller に統合している。

こちらでは、

- `scene_hint_bundle`
- `workspace_hint_bundle`
- `interaction_reasoning_hint_bundle`
- `interaction_audit_hint_bundle`

を flat payload として配線するのでなく、
controller が直接読む bundle として扱うのが正しい。

### 4. sparse input から valid manifold へ投影する

Codebook Matching は、
曖昧で疎な入力からでも、valid な motion manifold 側へ投影する発想を取る。

こちらでは、

- 少ないユーザー入力
- 不完全な turn 情報
- 微小な shared moment

から、いきなり wording を出すのでなく、
まず valid な reaction manifold に落としてから表出する、
という設計に対応する。

## いまの repo にどう写すか

### A. PFNN / DeepPhase から引くもの

導入候補:

- `interaction_phase_state.py`

役割:

- turn の entry / sustain / release を持つ
- `hold / backchannel / speak / defer` の timing を phase で補助する
- `joint_state` と `heartbeat_structure_state` を使って co-regulation の相を持つ

入れる場所:

- `inner_os`
- できれば `reaction-first` の近傍

### B. MANN から引くもの

導入候補:

- `response_mode_mixer.py`
- または `response_selection_state` の内部 mixer

役割:

- `organism_state`
- `joint_state`
- `terrain_dynamics`
- `interaction_phase_state`

から、複数の反応 expert を重み付きで混ぜる。

ねらい:

- `if` の増殖を避ける
- response channel の選択を state-driven にする

### C. Neural State Machine から引くもの

導入候補:

- `scene_interaction_controller.py`

役割:

- `scene_hint_bundle`
- `workspace_hint_bundle`
- `interaction_reasoning_hint_bundle`
- `interaction_audit_hint_bundle`

を読む controller を 1 箇所に置く。

ねらい:

- `runtime.py` の flat key consumer を減らす
- scene/workspace/reasoning/audit の束を controller 単位で扱う

### D. Codebook Matching から引くもの

導入候補:

- `reaction_codebook_state.py`
- あるいは `response_selection_state` の候補束

役割:

- sparse cue から候補反応群を作る
- その中から `terrain / joint / memory` に合う反応を選ぶ

ねらい:

- 曖昧入力に対して無理に 1 本へ潰さない
- prompt-only persona より continuity を優先する

## この repo での具体的な優先順

1. `interaction_phase_state.py` を追加する
2. `turn_timing_hint / response_channel / joint_state / heartbeat` から phase を導出する
3. `response_selection_state` に phase-conditioned な mixer を入れる
4. `scene_hint_bundle / workspace_hint_bundle / interaction_reasoning_hint_bundle / interaction_audit_hint_bundle` を読む controller を切る
5. continuity eval で prompt-only persona と比較する

## やらないこと

- AI4Animation をそのまま模倣して motion network 化する
- latent という言葉だけ借りて中身を曖昧にする
- phase を新しい mode 名の言い換えにする

## 要点

AI4Animation から引くべきなのは、

- 連続 phase
- 状態条件付き mixer
- scene interaction controller
- sparse control から valid manifold への投影

であって、
motion そのものではない。

こちらでは、
それを

- `memory`
- `joint`
- `terrain`
- `reaction-first`

へ翻訳して使うのが正しい。
