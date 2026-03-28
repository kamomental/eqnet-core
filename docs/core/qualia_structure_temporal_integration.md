# クオリア構造学と Temporal Memory Orchestration の統合方針

Date: 2026-03-24

## 目的

この文書は、`eqnet core / heartOS` の現在の設計を壊さずに、

- クオリア構造学
- 長期・時間依存の記憶検索 / 時系列整合

をどう統合するかを明文化するための設計メモである。

ここで重要なのは、`eqnet core` を捨てることではない。
むしろ、

- `記憶をどう整えるか`
- `整えた記憶がどう感じられるか`
- `その感じ方がどう残るか`

を分業させることで、今の core を強化する。

## 前提整理

まず、現在の `eqnet core` は最初から単純な「感情を点に置く」設計ではない。
すでに、

- 感情地形
- クオリア膜
- 接触
- 残響
- 可塑性
- carry / arc

という、場と変形を中心にした発想を持っている。

したがって、クオリア構造学を導入する時も、
「感情を点に落とし直す」のではなく、
**関係構造をより明示的に扱う**方向で使う。

## クオリア構造学から抽出する要点

### 1. 直接の中身ではなく、関係構造を扱う

クオリア構造学の核は、

- クオリアそのものを直接測ることは難しい
- しかし、クオリア同士の類似・非類似・近接・歪みは扱える

という立場にある。

`eqnet core` に引き直すと、

- `joy` というラベルを直接固定する

のではなく、

- 喜びの近傍に何があるか
- 何が混ざりやすいか
- どの方向へ漏れやすいか

を扱うべき、ということになる。

### 2. 構造は静的ではなく、注意・発達・文脈で変形する

クオリア構造学の研究では、

- 注意が外れると構造が崩れる
- 発達や文化をまたいでも一部は安定し、一部は変わる

ことが示唆されている。

`eqnet core` に引き直すと、

- 感情地図は固定辞書ではない
- `scene / body / relation / attention / memory` に応じて局所変形する

べきだと読める。

### 3. 重要なのはラベルでなく「にじみ」

クオリア構造学を AI キャラクタへ応用する時に重要なのは、

- 喜び
- 悲しみ
- 怒り

をそのまま出すことではない。

重要なのは、

- 喜びに近い皮肉
- 悲しみに近い共感
- 警戒に引かれた修復

のような**近傍の漏れ**である。

## temporal memory 側から抽出する要点

### 1. LongMemEval が示したこと

LongMemEval は、長期会話記憶の難所を

1. 情報抽出
2. 複数セッション推論
3. 時間推論
4. 更新知識
5. abstention

に分けている。

この分解はそのまま `Temporal Memory Orchestration` の責務になる。

さらに、LongMemEval の改善策として、

- session decomposition
- fact-augmented key expansion
- time-aware query expansion

が挙げられている。

つまり、長期記憶の本質は
「全部読むこと」ではなく、
**時間軸つきの再構成と検索条件の精密化**
にある。

### 2. LoCoMo が示したこと

LoCoMo は、長い会話の中で本当に難しいのは、

- 単なる再生ではなく
- 長距離の temporal / causal dynamics を理解すること

だと示している。

`eqnet core` に引き直すと、

- 単に「昔こう言っていた」を拾うだけでは不十分
- `その後どう変わったか`
- `何が上書きされたか`
- `今はどの場で再開すべきか`

まで扱う必要がある。

### 3. LOCCO が示したこと

LOCCO は、長期記憶には

- 記憶の経時減衰
- rehearsal の限界
- 記憶カテゴリごとの保持差

があることを示している。

これは `eqnet core` では、

- 何でも同じように残さない
- `carry_strength` と `decay`
- `identity / relation / unresolved / theme`
  で残り方を変える

べきだという圧になる。

## 統合の基本原則

### 原則 1: LLM を本体状態モデルにしない

これは従来どおり維持する。

- LLM は最後の表出層
- 記憶検索・感情地形・クオリア膜・carry は `inner_os` 側

### 原則 2: temporal は「記憶入力の整流」

`Temporal Memory Orchestration` の責務は、

- 何が最新か
- 何が過去のままか
- 何が訂正されたか
- 何が今の問いに時系列的に relevant か

を整理することに限定する。

### 原則 3: qualia は「感じ方の幾何」

クオリア構造側の責務は、

- その evidence がどう感じられるか
- どの感情近傍が励起されるか
- どの方向へ漏れるか

を決めることに限定する。

### 原則 4: readout は少数、場は豊かに

- `protection_mode`
- `commitment_state`
- `agenda_window_state`
- `social_topology_state`

のような readout は少数に保つ。

一方で下層では、

- 関係圧
- 身体負荷
- 未解決緊張
- 感情近傍の漏れ

を場として保持する。

## repo への対応

### 現在すでにあるもの

- 時間継続の最小核:
  - `inner_os/temporal.py`
- recall 境界:
  - `IntegrationHooks.memory_recall()`
- short-horizon continuity:
  - `working_memory_core`
- day / night summary:
  - `daily_carry_summary`
- 再開条件:
  - `agenda_window_state`
- 長期自己線:
  - `identity_arc`
  - `identity_arc_registry`
- 長期関係線:
  - `relation_arc`
  - `relation_arc_registry`
  - `group_relation_arc`
- 人格断片の選択:
  - `persona_memory_selector`

### まだ薄いもの

#### A. temporal evidence bundle

今の `memory_recall` は recall 前後の世界状態・identity trace・relation trace をかなり読むが、
次がまだ明示的ではない。

- `latest fact`
- `superseded fact`
- `timeline candidate`
- `temporal query intent`
- `same-group reentry evidence`
- `same-culture reentry evidence`

#### B. temporal query expansion

今は cue の再点火はあるが、

- `last time`
- `more recent`
- `before correction`
- `same session / another session`

のような時間演算を query plan として明示していない。

#### C. qualia structure map

今の `terrain` と `blend` はあるが、
クオリア構造学に対応する意味での

- 近傍
- 類似
- 漏れ
- 局所歪み

を明示的に持つ地図層はまだ薄い。

#### D. membrane operator の分離

`event + memory evidence + relation + body`
が、どう affective field に投影されるかを
独立した演算子としてまだ固定していない。

## 推奨アーキテクチャ

```text
Temporal Memory Orchestration
  ↓
MemoryEvidenceBundle
  ↓
Qualia Membrane Operator
  ↓
Qualia Structure Map / Affective Field
  ↓
Readout (protection / commitment / agenda / topology)
  ↓
Expression Bridge
  ↓
Post-turn / Nightly / Arc Carry
```

## 追加すべき最小モジュール

### 1. `memory_evidence_bundle.py`

責務:

- recall 候補のうち、時間整合済みの証拠束を持つ

最低限の中身:

- `facts_current`
- `facts_superseded`
- `timeline_events`
- `temporal_constraints`
- `reentry_context`
- `source_refs`

### 2. `temporal_memory_orchestration.py`

責務:

- `text_cue / world_cue / current_state / retrieval_summary`
  から
- `MemoryEvidenceBundle`
  を組む

主な仕事:

- session decomposition
- fact/update split
- latest-vs-stale resolution
- time-aware query expansion

### 3. `qualia_structure_map.py`

責務:

- affective labels の固定辞書ではなく、
  感情近傍と関係構造を持つ

主な中身:

- quality neighborhoods
- asymmetric spillover
- local deformation by relation/body/state

### 4. `qualia_membrane_operator.py`

責務:

- `event + MemoryEvidenceBundle + current_state`
  を `affective field perturbation` に写す

ここで初めて

- 何がどう感じられるか
- 何がどこへ漏れるか

を決める。

## 接続方針

### Phase 1: sidecar で導入

既存の `memory_recall()` を壊さず、

- まず `temporal_memory_orchestration`
  を sidecar として追加する
- 返り値に `memory_evidence_bundle` だけ足す

この段階では本流の意思決定を変えない。

### Phase 2: membrane へ入力

`memory_evidence_bundle` を、

- `terrain`
- `affect_blend`
- `qualia_planner_view`

の手前で使い、
affective field への投影を少し変える。

この段階でも hard safety は不変に保つ。

### Phase 3: readout へ波及

投影後の field を使って、

- `protection_mode`
- `commitment_state`
- `agenda_window_state`

が少し変わるようにする。

### Phase 4: arc/carry に残す

記憶事実だけでなく、

- どの近傍感情が縮んだか
- どの緊張が繰り返し再活性化したか
- どの relation で感じ方が変わったか

を `identity_arc / relation_arc / group_relation_arc` に残す。

## これで何が改善するか

### 1. temporal の改善

- 古い事実と新しい事実を混ぜにくくなる
- 「前回」「最新」「前の会合」の再開が扱いやすくなる
- public/private/group/culture ごとの再開条件が作りやすくなる

### 2. qualia の改善

- 感情が単独ラベルでなく近傍つきで出る
- 人格差が属性リストでなく、感情構造の歪みとして出る
- 同じ出来事でも relation/body/state に応じてにじみ方が変わる

### 3. expression の改善

- 定型返しが減る
- 「今この場での感じ方」が返答骨格に出やすくなる
- 保留と再開の自然さが上がる

## NG

- `Temporal Memory Orchestration` を単なるベクトル検索の言い換えにする
- qualia structure を固定 emotion point の辞書にする
- クオリア膜を比喩のまま放置する
- raw 会話ログをそのまま LLM に流して temporal を代用する
- `identity / relation / agenda` の readout を増やしすぎて state machine 化する

## 実装優先順位

1. `memory_evidence_bundle.py`
2. `temporal_memory_orchestration.py`
3. `qualia_membrane_operator.py`
4. `qualia_structure_map.py`
5. `memory_recall` への sidecar 接続

## 一番短い結論

クオリア構造学は `eqnet core` を置き換えない。
`eqnet core` の中で、

- `Temporal Memory Orchestration`
  - 何を思い出すか
- `Qualia Membrane / Structure`
  - どう感じに写すか
- `Arc / Carry`
  - 何が残るか

を分業させるための、かなり強い補強原理である。

## References

- [QUALIA STRUCTURE 公式サイト](https://en.qualia-structure.jp/)
- [A01: クオリア構造の実験心理学と数理](https://qualia-structure.jp/research/detail/412)
- [Enriched category as a model of qualia structure based on similarity judgements (Tsuchiya et al., 2022)](https://pubmed.ncbi.nlm.nih.gov/35436717/)
- [Comparing color qualia structures through a similarity task in young children versus adults (2025)](https://pubmed.ncbi.nlm.nih.gov/40067901/)
- [Qualia structures collapse for geometric shapes, but not faces, when spatial attention is withdrawn (2025)](https://pubmed.ncbi.nlm.nih.gov/41404383/)
- [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d813d324dbf0598bbdc9c8e79740ed01-Abstract-Conference.html)
- [Evaluating Very Long-Term Conversational Memory of LLM Agents / LoCoMo (ACL 2024)](https://aclanthology.org/2024.acl-long.747/)
- [Evaluating the Long-Term Memory of Large Language Models / LOCCO (Findings ACL 2025)](https://aclanthology.org/2025.findings-acl.1014/)
