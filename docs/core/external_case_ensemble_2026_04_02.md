# External Case Ensemble (2026-04-02)

## 目的

`memory_dynamics / joint_state / terrain_dynamics / llm_hub` を再構成するにあたって、
外部事例のうち **そのまま import すべきでないが、設計原理として有効なもの** を整理する。

この文書の立場は次の通り。

- 既存 repo の断片を捨てて別方式へ移行するのではない
- 外部事例から **分業の切り方** を学び、`inner_os` の canonical core に写す
- LLM を本体記憶にしない
- graph を保存物としてではなく、**発火・比較・再固定化の場** として使う

## 確認した外部事例

### 1. GraphRAG

出典:

- [GraphRAG docs](https://microsoft.github.io/graphrag/)
- [GraphRAG outputs](https://microsoft.github.io/graphrag/index/outputs/)
- [GraphRAG query overview](https://microsoft.github.io/graphrag/query/overview/)

確認できた要点:

- entity / relationship / claim を抽出して graph を作る
- community hierarchy を bottom-up に要約する
- query 時に `Global / Local / DRIFT / Basic` を使い分ける

この repo への写像:

- 採る:
  - `palace_topology`
  - hierarchical summary
  - `global/local/drift` 的な複数検索視点
- 採らない:
  - graph を static retrieval の正本にすること
- 役割:
  - **静的構造と多粒度 summary の層**

### 2. HippoRAG

出典:

- [HippoRAG paper page (arXiv 2405.14831)](https://huggingface.co/papers/2405.14831)

確認できた要点:

- knowledge graph と Personalized PageRank を組み合わせる
- hippocampal indexing 的に、query 時の co-activation を重視する
- iterative retrieval より軽く multi-hop を取る

この repo への写像:

- 採る:
  - `ignition_readiness`
  - query-time co-activation
  - graph 上の伝播
- 採らない:
  - retrieval を QA 最適化だけで考えること
- 役割:
  - **動的発火と traversal の層**

### 3. REMem

出典:

- [REMem (ICLR 2026 OpenReview)](https://openreview.net/forum?id=fugnQxbvMm)

確認できた要点:

- offline indexing で episodic memory を hybrid memory graph に変換する
- time-aware gists と facts を分けて持つ
- online inference で iterative retrieval を行う

この repo への写像:

- 採る:
  - gist / fact の二層
  - time-aware episodic graph
  - offline indexing と online recollection の分離
- 採らない:
  - エピソードを retrieval 専用の外部 DB としてのみ扱うこと
- 役割:
  - **episodic graph と reconsolidation の層**

### 4. Common Ground / Cooperative Communication

出典:

- [A World Unto Itself: Human Communication as Active Inference](https://pmc.ncbi.nlm.nih.gov/articles/PMC7109408/)

確認できた要点:

- communication は共同で mental state を揃える過程
- `common ground` が増えるほど必要な communication は減る
- communication は circular / bidirectional flow を持つ

この repo への写像:

- 採る:
  - `joint_state`
  - `common_ground`
  - `repair_readiness`
  - `shared_delight`
- 採らない:
  - self state だけで response を決めること
- 役割:
  - **self-other coupling と場の共有状態の層**

### 5. Dynamic Knowledge Graph の運用例

出典:

- [Nature Communications: dynamic knowledge graph approach](https://www.nature.com/articles/s41467-023-44599-9)

確認できた要点:

- knowledge graph は agent が更新し続ける運用対象として扱える
- provenance と execution trace を graph 側へ戻す
- graph は静的 index ではなく evolving system になりうる

この repo への写像:

- 採る:
  - provenance
  - update trace
  - evolving graph
- 採らない:
  - 全更新を graph だけに寄せること
- 役割:
  - **carry / nightly / audit と整合する運用層**

## アンサンブル方針

### A. `memory_dynamics` は 1 本に束ねる

外部事例を踏まえると、`memory_dynamics` は最低でも次の 4 層に分けるのが自然。

1. `topology`
- palace
- community
- gist cluster

2. `relation field`
- relation edge
- edge hypothesis
- edge confidence
- edge decay / reinforcement

3. `activation`
- ignition
- traversal
- co-activation
- barrier crossing

4. `reconsolidation`
- monument reinforcement
- gist rewrite
- forgetting pressure
- carry

### B. `joint_state` は retrieval 条件でもある

GraphRAG 系は graph 側に重心があるが、この repo では `joint_state` を抜くと不自然になる。

したがって relation retrieval / generation は、

- `organism_state`
- `external_field_state`
- `joint_state`
- `terrain_dynamics`

に依存して変わるべきである。

つまり relation は static edge ではなく、
**state-indexed edge hypothesis** として扱う。

### C. `relation` と `meta-relation` を first-class にする

今後の canonical 最小単位は次の 2 つ。

1. `relation edge`
- episode と episode
- episode と theme
- theme と theme
- shared moment と prior thread

2. `meta-relation`
- reinforces
- competes_with
- generalizes
- specializes
- gated_by
- resolved_by

ここで初めて

- `あの時と今日が同じ問題`
- `前回の repair が今回の shared delight を開いた`
- `この笑いは以前の strained jointness を少し緩めた`

を state として持てる。

### D. `llm_hub` は graph narrator ではなく expression bridge

外部事例は retrieval と reasoning を graph へ寄せるが、この repo では
`llm_hub` を本体に戻してはいけない。

したがって raw LLM に渡すのは:

- dominant relation
- current meta-relation
- utterance reason
- joint mode
- discourse shape

までに圧縮する。

graph 全体や raw observation をそのまま流さない。

## この repo で次にやること

### 1. `memory_dynamics` に relation layer を追加する

最低限:

- `MemoryRelationEdge`
- `MemoryMetaRelation`
- `relation_generation_mode`
- `dominant_relation_type`
- `dominant_relation_key`

### 2. `association_graph` を relation source の 1 つに下ろす

`association_graph` はそのまま使うが、canonical truth ではなく
`memory_dynamics` へ流し込む source にする。

### 3. `joint_state` を relation generation に接続する

relation の生成・強化・抑制に

- shared tension
- shared delight
- common ground
- repair readiness

を直接使う。

### 4. `utterance_reason` は dominant relation から作る

今後の `why this utterance` は:

- self 状態だけ
- moment kind だけ

ではなく、

- current dominant relation
- current meta-relation
- current joint mode

を含むようにする。

## 採用しないこと

- GraphRAG をそのまま core に据える
- static graph を正本にする
- relation を保存だけして生成しない
- LLM に relation construction を丸投げする

## 要約

外部事例を総合すると、この repo の次の本命は

- `memory_dynamics = topology + relation field + activation + reconsolidation`
- `joint_state = common ground を含む relation conditioning`
- `llm_hub = relation-aware expression bridge`

として再構成すること。

GraphRAG は構造、
HippoRAG は発火、
REMem はエピソード時間、
common ground 論は jointness、
dynamic KG 運用は provenance と更新規律、
として取り込む。
