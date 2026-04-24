# Repo Fragment Integration Map

## 目的

この文書は、`emotional_dft` 全体に散っている

- terrain / field / entropy / ignition
- palace / monument / recall
- growth / forgetting / reconsolidation
- qualia / heartbeat / projection
- runtime / live / probe

の断片を、どこで canonical core に束ねるべきかを整理するための map です。

ここで重要なのは、**断片が無いことが問題なのではなく、断片が複数の package に散っていて、まだ 1 つの organism dynamics として束ね切れていないこと**です。

## 現在の大きな分担

| 層 | 主な置き場 | 役割 |
|---|---|---|
| 場・地形・熱力学 | `emotional_dft`, `eqnet_core`, `emot_terrain_lab/terrain`, `devlife` | field / terrain / entropy / ignition / control の仮説本体 |
| 実行・統合・live | `emot_terrain_lab` | runtime, LM Studio, VLM/TTS, probe, ops |
| 再利用可能な状態核 | `inner_os` | continuity, memory contract, growth, epistemic, expression contract |
| 知識・履歴・文化 | `memory`, `persona`, `culture`, `rag` | 長期知識、関係、文化、説明資料 |

## すでに存在する重要断片

### 1. palace / monument / ignition

| 概念 | 既存断片 | いま担っている役割 |
|---|---|---|
| palace | [memory_palace.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/emot_terrain_lab/terrain/memory_palace.py) | locale ごとの node・trace・qualia state を持つ記憶配置 |
| monument | [memory_orchestration_core.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/memory_orchestration_core.py) | `monument_salience` `consolidation_priority` `reuse_trajectory` |
| ignition | [recall_engine.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/emot_terrain_lab/hub/recall_engine.py) | `ignite()` による anchor cue からの activation chain |
| associative topology | [association_graph.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/association_graph.py) | seed 間リンク、prior、novelty、未解決性の連結 |

### 2. terrain / energy / entropy / ignition index

| 概念 | 既存断片 | いま担っている役割 |
|---|---|---|
| emotion field | [architecture.md](/C:/Users/kouic/Desktop/python_work2/emotional_dft/emot_terrain_lab/architecture.md) / [system.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/emot_terrain_lab/terrain/system.py) | emotional field, membrane, diary, story graph, control の統合 |
| entropy / ignition threshold | [loop.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/devlife/runtime/loop.py) | `ignite_delta_R_thresh`, `ignite_entropy_z_thresh`, `ignition_index` |
| external field metrics | [loop.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/devlife/runtime/loop.py) | `_compute_field_metrics`, `_merge_external_field_metrics`, `_pull_field_metrics` |
| attractor / replay loop | [system.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/emot_terrain_lab/terrain/system.py) | field, catalyst, story graph, rest-state, MemoryPalace の統合運用 |

### 3. projection / observable

| 概念 | 既存断片 | いま担っている役割 |
|---|---|---|
| qualia projection | [qualia_projector.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/qualia_projector.py) | latent state から `gate / qualia / precision / observability / body_coupling` を投影 |
| qualia temporal structure | [qualia_structure_state.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/qualia_structure_state.py) | `phase / emergence / stability / drift / trace` |
| heartbeat temporal structure | [heartbeat_structure_state.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/heartbeat_structure_state.py) | `pulse_band / containment_bias / bounce_room / dominant_reaction / trace` |

### 4. slow-state / development

| 概念 | 既存断片 | いま担っている役割 |
|---|---|---|
| growth | [growth_state.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/growth_state.py) | `relational_trust / expressive_range / playfulness_range / self_coherence` |
| development transition | [development_transition_policy.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/development_transition_policy.py) | development / forgetting / consolidation / transfer の統合更新 |
| epistemic maturity | [epistemic_state.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/epistemic_state.py) | `freshness / verification_pressure / stale_risk / epistemic_caution` |
| forgetting | [forgetting_core.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/forgetting_core.py) | pressure / replay horizon / half-life |
| reconsolidation | [sleep_consolidation_core.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/sleep_consolidation_core.py) | replay / reconsolidation / abstraction readiness |

### 5. 会話理由 / expression bridge

| 概念 | 既存断片 | いま担っている役割 |
|---|---|---|
| shared moment | [shared_moment_state.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/shared_moment_state.py) | 小出来事の型 |
| appraisal | [appraisal_state.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/appraisal_state.py) | 現在の出来事をどう評価したか |
| meaning update | [meaning_update_state.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/meaning_update_state.py) | 今回何が少し変わったか |
| utterance reason | [utterance_reason_packet.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/utterance_reason_packet.py) | だから何を返すか |
| discourse shape | [discourse_shape.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/expression/discourse_shape.py) | `act -> discourse shape` の typed contract |
| runtime shaping | [runtime.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/emot_terrain_lab/hub/runtime.py) | final shaping, compact, qualia gate, live loop |

## いまの本質的な問題

### 1. 断片はあるが 1 つの力学になっていない

いまある state は多いが、

- `growth`
- `epistemic`
- `qualia`
- `heartbeat`
- `relation`

がまだ **同じ organism dynamics の射影** になっていません。

### 2. palace / monument / ignition も 1 本化されていない

- palace は `terrain/memory_palace.py`
- monument salience は `inner_os/memory_orchestration_core.py`
- ignition は `emot_terrain_lab/hub/recall_engine.py`
- association は `inner_os/association_graph.py`

に分かれていて、まだ **memory_dynamics** として統一されていません。

### 3. terrain / energy / entropy は `devlife` 側に強い断片がある

`devlife/runtime/loop.py` には

- ignition thresholds
- entropy / delta_R
- field metrics merge
- valence / arousal projection

があり、これは単なる補助実験ではなく、**terrain dynamics の source fragment** と見るべきです。

### 4. 表出は canonical 化が進んだが、まだ upstream と完全一致していない

`appraisal -> meaning update -> utterance reason -> discourse shape`
までは `inner_os` に入ったが、

- upstream route
- live engagement
- runtime legacy containment path

とはまだ完全には一致していません。

## canonical に昇格させるべきもの

### A. `organism_state`

**目的**: `heartbeat / qualia / growth / epistemic / relation` を 1 つの小さい latent organism state に束ねる。

**既存断片の入力元**
- [growth_state.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/growth_state.py)
- [epistemic_state.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/epistemic_state.py)
- [qualia_structure_state.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/qualia_structure_state.py)
- [heartbeat_structure_state.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/heartbeat_structure_state.py)
- relation / thread 系 state

**昇格理由**
- state の数を増やすのでなく、同じ organism の view に戻すため

### B. `memory_dynamics`

**目的**: `palace / monument / ignition / reconsolidation / forgetting` を 1 本の memory dynamics に束ねる。

**既存断片の入力元**
- [memory_palace.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/emot_terrain_lab/terrain/memory_palace.py)
- [memory_orchestration_core.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/memory_orchestration_core.py)
- [association_graph.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/association_graph.py)
- [recall_engine.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/emot_terrain_lab/hub/recall_engine.py)
- [forgetting_core.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/forgetting_core.py)
- [sleep_consolidation_core.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/sleep_consolidation_core.py)

**昇格理由**
- 記憶を保存物ではなく、再活性化と再固定化の力学として扱うため

### C. `external_field_state`

**目的**: 入力文だけでなく、場・履歴・関係・ノイズ・live の外力を 1 つの external field として扱う。

**既存断片の入力元**
- [loop.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/devlife/runtime/loop.py)
- `runtime` の live / probe / controls
- relation / thread / group / culture 系 state

**昇格理由**
- `external != input`
- 外部は境界条件・外力・ノイズとして state に影響するため

### D. `projection_observables`

**目的**: valence / arousal / pulse / posture を state そのものではなく projection として明示する。

**既存断片の入力元**
- [qualia_projector.py](/C:/Users/kouic/Desktop/python_work2/emotional_dft/inner_os/qualia_projector.py)
- `devlife` の valence / arousal projection
- heartbeat / qualia packet axes

**昇格理由**
- パラメータを主役にせず、状態の観測インターフェースに戻すため

## `inner_os` に入れるもの / 残すもの

### `inner_os` に入れる

- typed な canonical state
- typed な transition policy
- typed な expression contract
- state を projection へ落とす boundary

### `emot_terrain_lab` に残す

- runtime loop
- LM Studio / live probe
- VLM / TTS / UI
- final shaping の adapter

### `terrain` / `devlife` に残す

- field / terrain / thermodynamics 仮説
- entropy / ignition / external field metrics の研究線
- 低レベルの身体・発火・alert の実験

## 次の統合順

1. `AMA-07`
   - palace / monument / ignition を `memory_dynamics` の source fragment として整理する
2. `AMA-11`
   - `growth / epistemic / qualia / heartbeat / relation` を `organism_state` に束ねる
3. `AMA-12`
   - `joint_state` を作り、会話を共同作用として扱う
4. `AMA-13`
   - `memory_dynamics` を canonical 化する
5. `AMA-14`
   - explorer を static distance から temporal state viewer へ変える
6. `AMA-15`
   - VLM / TTS を含む closed loop へつなぐ

## 一言でいうと

この repo は、必要なものが無いのではなく、**必要なものがすでに複数箇所に存在している段階**です。

次の勝負は新発明ではなく、

- 何を canonical core に昇格させるか
- 何を runtime / terrain / devlife の adapter として残すか

を間違えずに束ねることです。
