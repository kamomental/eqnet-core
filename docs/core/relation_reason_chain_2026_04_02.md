# Relation Reason Chain (2026-04-02)

## 概要

`memory_dynamics` に relation field を入れた段階では、まだ
「何を覚えているか」が中心だった。

今回の狙いは、そこから一歩進めて、
**「今どの relation / causal link が立ち上がっているか」**
が `appraisal -> meaning_update -> utterance_reason` を決めるようにすることにある。

つまり、記憶を保存物として使うのでなく、
**このターンで前景化した関係と因果を response cause に変換する**
ための chain である。

## 現在の chain

### 1. memory_dynamics

`memory_dynamics` は canonical な relation / causal field として扱う。

主に使うもの:

- `dominant_relation_type`
- `relation_generation_mode`
- `relation_edges`
- `meta_relations`
- `dominant_causal_type`
- `causal_generation_mode`
- `causal_edges`

source は `association_graph` や `memory_mosaic` だが、
reason chain が直接読むのは `memory_dynamics` のみとする。

### 2. appraisal

`appraisal` は、今の shared moment や memory resonance を
relation / causal hypothesis と一緒に読む。

現在の主入力:

- `dominant_relation_type`
- `dominant_relation_key`
- `relation_meta_type`
- `dominant_causal_type`
- `dominant_causal_key`

ここでは
「何が起きたか」だけでなく
**何がその変化を起こしやすくしたか**
まで扱う。

### 3. meaning_update

`meaning_update` は appraisal を relation / causal frame に落とす。

relation 側の frame:

- `same_anchor_link`
- `unfinished_link`
- `cross_context_bridge`
- `returning_pattern`

causal 側の frame:

- `same_anchor_cause`
- `unfinished_thread_cause`
- `reframing_cause`
- `memory_trigger_cause`
- `reinforced_cause`
- `guarded_cause`

これで、同じ smile や relief でも
「既知の thread 上で enabled された笑い」
「unfinished carry が reopened された笑い」
のように意味が変わる。

### 4. utterance_reason

`utterance_reason` は次を主入力に持つ。

- `relation_frame`
- `relation_key`
- `causal_frame`
- `causal_key`
- `memory_frame`
- `memory_anchor`

ここで決まるのは、
単に「small laugh があった」ではなく、
**どの relation / causal link を保ちながら返すか**
である。

### 5. joint_state

`joint_state` は shared moment だけでなく、
memory 由来の relation / causal field でも変化する。

現在 relation-aware / causal-aware なのは次の軸:

- `shared_delight`
- `shared_tension`
- `repair_readiness`
- `common_ground`
- `joint_attention`
- `coupling_strength`

これにより common ground は、
単なる moment 共有ではなく
**今この場で共有できた relation / causal link**
に基づいて更新される。

## 設計原則

- LLM を relation / causal 推論の本体にしない
- `memory_dynamics` を canonical source にする
- `appraisal -> meaning_update -> utterance_reason` は解釈層として保つ
- response は mode からではなく reason chain から決める

## 次の本命

次に進めるべきは次の 3 本。

1. `terrain_dynamics` を relation-conditioned / causal-conditioned にする
2. `route / response_strategy / action_posture` を同じ chain から決める
3. `hold / backchannel / nonverbal response` まで同じ reason chain を通す

ここまで進むと、
「何を覚えているか」ではなく
**「今どの関係と因果が立ち上がったか」から反応が決まる**
ところまで上がる。
