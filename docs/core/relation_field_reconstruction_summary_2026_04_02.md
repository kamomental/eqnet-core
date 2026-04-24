# Relation Field Reconstruction Summary (2026-04-02)

## 目的

`memory_dynamics` の relation field を、

- `appraisal`
- `meaning_update`
- `utterance_reason`
- `joint_state`

の主語に昇格させる。

ここでの方針は、
「何を覚えているか」ではなく
「今どの relation が立ち上がっているか」
で反応が決まるようにすること。

## まず固定した原則

- LLM を本体にしない
- 感情をラベルにしない
- mode を先に選ばない
- 発話を先に決めない
- state と dynamics を正本にする

## 今回の判断

### 1. `memory_dynamics` は relation field を持つ canonical state にする

最低限の正本:

- `relation_edges`
- `meta_relations`
- `dominant_relation_type`
- `relation_generation_mode`

relation の立場:

- `association_graph` は source
- `memory_dynamics` は canonical relation field
- relation は static graph の保存物ではなく、今回の想起で前景化した仮説

### 2. `appraisal` は relation-aware にする

今回の appraisal は、

- `dominant_relation_type`
- `dominant_relation_key`
- `relation_meta_type`

を受け取る。

ここで relation は、背景の strain / remembered thread / easing shift を変える条件として使う。

### 3. `meaning_update` は relation frame を持つ

最小の frame:

- `same_anchor_link`
- `unfinished_link`
- `cross_context_bridge`
- `returning_pattern`

これで、

- 単に shared smile があった
- 既知 thread が近い

だけでなく、

- 今回は同じ anchor の問題が戻った
- 未完了の link がまた動いた
- 遠い文脈が橋渡しされた

を state として持てる。

### 4. `utterance_reason` は relation を response cause に昇格させる

`utterance_reason_packet` は今後、

- `reason_frame`
- `relation_frame`
- `relation_key`
- `memory_frame`

を持つ。

つまり、
「small laugh に反応する」
だけでなく、
「same_anchor_link が立っているので brief_shared_smile を返す」
まで cause を持つ。

### 5. `joint_state` も relation field に条件づける

`joint_state` は shared moment だけでなく、

- relation type
- relation meta
- relation generation mode

を使って

- `shared_delight`
- `shared_tension`
- `repair_readiness`
- `common_ground`
- `joint_attention`
- `coupling_strength`

を少しずつ更新する。

これで common ground は「雰囲気」ではなく、
relation field に裏打ちされた shared state になる。

## 外部事例から採ったもの

### GraphRAG

- 採る: static structure と multi-granular summary
- 採らない: static graph を正本にすること

### HippoRAG

- 採る: query-time co-activation と traversal
- 採らない: retrieval を QA 用最適化で閉じること

### REMem

- 採る: time-aware episodic graph と gist / fact の分離
- 採らない: episodic memory を単なる外部 DB にすること

### Common Ground / Active Inference

- 採る: communication を shared state 更新として見る視点
- 採らない: self state だけで response を決めること

### Feed-forward 3D reconstruction の内部幾何論文

出典:

- [arXiv:2512.11508](https://arxiv.org/abs/2512.11508)

ここから受けた圧力:

- 「内部構造がある」だけでは独自性にならない
- eqnet core は
  - 持続性
  - 可塑性
  - 履歴依存性
  - 熱力学性
  - 価値勾配
  - 共同性
  を持つ dynamics として厳密化する必要がある

## 今回の実装要点

- `memory_dynamics` に relation field を追加
- `appraisal / meaning_update / utterance_reason / joint_state` を relation-aware 化
- `surface_context_packet` に relation-aware source state を追加
- `llm_hub` の `response_cause` に relation frame / key を流す

## いまの位置づけ

ここまではまだ
「relation-aware reason chain」
であって、
`terrain_dynamics` の basin / barrier / recovery 自体が relation-conditioned になった段階ではない。

つまり現在地は:

- 記憶 relation が reason chain に入った
- まだ route / strategy / terrain center までは完全移行していない

## 次の本命

1. `terrain_dynamics` を relation-conditioned にする
- basin
- barrier
- ignition
- recovery

2. `route / response_strategy / action_posture` を relation-aware にする

3. `nonverbal / hold / waiting` を正式行動にする
- speak
- backchannel
- filler
- hold
- defer
- silence / tempo

## まとめ

今回の再構成で大事なのは、
memory を「保存された過去」から
**「今この場で立ち上がる relation field」**
へ反転したこと。

次はこの relation field を、
reason chain だけでなく terrain と route の本体に上げる。
