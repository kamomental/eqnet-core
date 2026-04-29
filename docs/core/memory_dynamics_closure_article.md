# 記憶は検索箱ではなく、返答を形づくる地形である

副題: memory palace / monument / ignition / closure_packet のつながり

この記事は、EQNet の現在のリポジトリ実装を前提に、`memory_dynamics`、メモリーパレス、モニュメント、発火状態、再固定化、忘却、`ClosurePacket` がどうつながるかを説明します。

## まず全体像

EQNet では、記憶は「過去の文章を検索して返す箱」ではありません。

記憶は、今の会話に対して、

```text
何が立ち上がりやすいか
何を言いすぎない方がよいか
どの距離感で返すべきか
どの記憶を今は使わない方がよいか
どの話題は戻れるように残すべきか
```

を変える、slow-state の地形です。

現在の実装では、その中心に `memory_dynamics` があります。

```text
memory_palace
monument
association_graph
recall / activation_trace
forgetting
sleep_consolidation
        ↓
memory_dynamics
        ↓
joint_state
        ↓
closure_packet
        ↓
reaction_contract
        ↓
返答
```

重要なのは、`memory_dynamics` が直接返答文を作らないことです。

`memory_dynamics` は、記憶の状態を返答制御に使える軸へまとめます。
`ClosurePacket` は、その軸を「今回の返答で使える根拠・制約・不確かさ」へ変換します。
`reaction_contract` が、最終的に話すか、黙るか、短くするか、質問しないかを決めます。

## memory_dynamics が束ねているもの

コード上では、`inner_os/memory_dynamics.py` の `MemoryDynamicsState` が中心です。

主な軸は次です。

```text
palace_topology
palace_density
palace_mode

monument_salience
monument_kind
monument_mode

ignition_readiness
ignition_mode
activation_confidence
recall_anchor

consolidation_pull
replay_priority
reconsolidation_priority
autobiographical_pull
reconsolidation_mode

forgetting_pressure
memory_tension
prospective_pull

dominant_relation_type
dominant_causal_type
dominant_mode
```

これらは、記憶の内容そのものではありません。

たとえば、

```text
「あの日、港で約束した」
```

という記憶内容を保存するのが `memory_dynamics` の主目的ではありません。

`memory_dynamics` が扱うのは、次のような状態です。

```text
その記憶に近い連想が今どれだけまとまっているか
その記憶がモニュメントとしてどれだけ強いか
再想起が発火しそうか
再固定化すべきか
忘却圧や干渉圧が高いか
今回の返答に引き込むには危険か
```

## メモリーパレス: 記憶の地図

メモリーパレスは、記憶の地図です。

実装では `emot_terrain_lab/terrain/memory_palace.py` にあり、`MemoryNode` と `MemoryPalace` が、場所ごとの trace、label、qualia_state を持ちます。

ざっくり言うと、メモリーパレスは次を持っています。

```text
どのノードがあるか
どのノードに trace が溜まっているか
どの label が紐づいているか
qualia_state の memory / energy / magnitude などがどう変化したか
```

`memory_dynamics` は、メモリーパレスをそのまま返答に出しません。

代わりに、次のような圧縮値として読みます。

```text
active_density
  どれくらいのノードが動いているか

dominant_node
  いちばん強く立っているノード

dominant_load
  そのノードの負荷や存在感

qualia_memory
  qualia_state の memory 成分

topology_hint
  記憶地図がどれくらいまとまっているか
```

これが `palace_density` や `palace_topology` に入ります。

一般語で言うと、

```text
メモリーパレスは「どの記憶の場所が光っているか」を見る。
memory_dynamics は「その光り方が、今の会話にどの程度関係するか」を数値化する。
```

## モニュメント: 長く残る意味のアンカー

モニュメントは、一回きりの記憶ではなく、長く残る意味のアンカーです。

実装上は複数の入口があります。

- `inner_os/monument_query_adapter.py`
- `eqnet/memory/monuments.py`
- culture / place / partner / emotion による monument query
- `MemoryOrchestrationCore` の `monument_salience / monument_kind`

モニュメントは、たとえば次のようなものです。

```text
いつも戻ってくる場所
相手との関係を象徴する出来事
繰り返し出てくる約束や未解決の話題
その人らしさに関わる長期テーマ
```

`memory_dynamics` では、主に次の軸になります。

```text
monument_salience
  今そのモニュメントがどれくらい強く効いているか

monument_kind
  どんな種類のモニュメントか

monument_mode
  ambient / tagged / rising / engraved
```

`monument_mode` は、強さによって変わります。

```text
ambient
  まだ特に強くない

tagged
  種類は見えている

rising
  今の会話で立ち上がってきている

engraved
  強く刻まれたアンカーとして効いている
```

モニュメントが強いと、`ignition_readiness` や `consolidation_pull` にも影響します。

つまり、

```text
モニュメントが強い
  -> 思い出されやすい
  -> 再固定化されやすい
  -> 返答に慎重さや連続性が出る
```

という流れです。

## association_graph: 記憶同士のつながり

`association_graph` は、DotSeed 同士のリンクを作ります。

たとえば、

```text
harbor
promise
wind
```

という seed があると、

```text
harbor -> promise
promise -> wind
```

のような link が作られます。

link には次のような理由が付きます。

```text
anchor_overlap
source_diversity
novelty_gain
unresolved_relief
association_memory
```

`memory_dynamics` はこの link から、さらに関係と因果を作ります。

```text
relation_edges
  same_anchor
  unfinished_carry
  cross_context_bridge
  recurrent_association
  association_bridge

causal_edges
  enabled_by
  reopened_by
  reframed_by
  triggered_by
  amplified_by
  suppressed_by
```

ここが重要です。

`association_graph` は「似ている記憶を探す」だけではありません。
`memory_dynamics` に入ると、「その連想が今回の返答をどう動かすか」という relation / causal field になります。

## 発火状態: 今まさに立ち上がるか

発火状態は、記憶や連想が今まさに返答に影響しそうかを見る軸です。

関連する主な値は次です。

```text
ignition_readiness
ignition_mode
activation_confidence
recall_anchor
```

`recall_engine` は、anchor cue、seed、activation_chain、confidence_curve、replay_events を使って、発火の trace を作ります。

`memory_dynamics` はそこから、

```text
anchor_hit
chain_strength
internal_confidence
external_confidence
anchor_confirm
```

を読みます。

そして `ignition_mode` を決めます。

```text
idle
  まだ立ち上がっていない

arming
  少し準備されている

primed
  かなり立ち上がりやすい

active
  すでに発火している
```

たとえば、相手がこう言ったとします。

```text
この前の話、まだ少し残ってる。
```

この入力が、過去の未解決トピックと強くつながる場合、

```text
recall_anchor: previous_unfinished_thread
ignition_mode: primed または active
dominant_relation_type: unfinished_carry
dominant_causal_type: reopened_by
```

のようになります。

ただし、ここでも大事なのは、発火したからといって必ず話すわけではないことです。

発火していても、

```text
memory_tension が高い
basis_confidence が低い
reconstruction_risk が高い
```

なら、`ClosurePacket` はむしろ「言いすぎるな」という制約を作ります。

## 再固定化と睡眠: 何を残し直すか

`sleep_consolidation_core.py` は、夜間や長期運用で何を再生・再固定化・抽象化するかを決めるための snapshot を作ります。

主な軸は次です。

```text
replay_priority
reconsolidation_priority
autobiographical_pull
abstraction_readiness
```

`memory_dynamics` では、これらが次に入ります。

```text
consolidation_pull
replay_priority
reconsolidation_priority
autobiographical_pull
reconsolidation_mode
```

`reconsolidation_mode` は次のように分かれます。

```text
settle
  いったん落ち着かせる

replaying
  再生・再想起に寄る

reconsolidating
  意味や関係を固定し直す

defragmenting
  干渉や忘却圧が高く、整理が必要
```

会話中の一回の返答だけでなく、夜間の再整理が次の日の `memory_dynamics` に影響します。

つまり、

```text
今日の会話
  -> memory_dynamics
  -> sleep_consolidation
  -> 次の日の memory_dynamics
```

というループがあります。

## 忘却圧: 使わないための力

忘却は、単に記憶を消すことではありません。

`forgetting_core.py` では、stress、recovery_need、terrain_transition_roughness、transition_intensity、recent_strain から `forgetting_pressure` を作ります。

`forgetting_pressure` が高いと、`memory_dynamics` では次に効きます。

```text
ignition_readiness を下げる
memory_tension を上げる
reconsolidation_mode を defragmenting に寄せる
```

つまり、忘却圧は「今それを使わない方がいい」という保護信号でもあります。

たとえば、

```text
疲れている
場が荒れている
不確かさが高い
過去の記憶が混線しやすい
```

とき、忘却圧は上がります。

その結果、

```text
記憶を強く引っ張らない
断定しない
短く受ける
今は再構成しない
```

という返答制約につながります。

## dominant_mode: 今の記憶状態を一語で見る

`memory_dynamics` は最後に `dominant_mode` を決めます。

```text
ignite
  記憶や連想が発火している

reconsolidate
  再生・再固定化が主になっている

prospect
  未来側への引き込みが強い

protect
  緊張や干渉が高く、保護が主になっている

stabilize
  安定化が主になっている
```

これは返答の最終判断ではありません。

しかし、`joint_state`、`ClosurePacket`、`reaction_contract` にとって、かなり重要な上流信号です。

## ClosurePacket への接続

`ClosurePacket` は、`memory_dynamics` を次のように読みます。

```text
dominant_basis_keys
  dominant_link_key
  recall_anchor
  dominant_relation_type
  dominant_causal_type

basis_confidence
  activation_confidence
  monument_salience
  association weight
  common_ground

closure_tension
  memory_tension
  shared_tension
  unknown_likelihood
  boundary instability

reconstruction_risk
  memory_tension
  low basis confidence
  unknown attribution
  forgetting pressure
```

そして、次のような返答制約を作ります。

```text
generated_constraints:
  - do_not_overinterpret
  - leave_return_point
  - do_not_reconstruct_memory
  - preserve_boundary

generated_affordances:
  - shared_anchor
  - gentle_join
  - repair_window

inhibition_reasons:
  - memory_tension
  - reconstruction_risk
  - unknown_attribution

uncertainty_reasons:
  - low_basis_confidence
  - weak_association_margin
```

つまり `ClosurePacket` は、

```text
記憶が今どう動いているか
```

を、

```text
今回の返答では何を言いすぎない方がよいか
どこまで近づいてよいか
何を不確かさつきで扱うべきか
どの返り道を残すべきか
```

へ変換します。

## 例: 「今日はちょっと無理かも」

相手:

```text
今日はちょっと無理かも。
```

内部では、次のようなことが起きます。

```text
memory_palace:
  似た緊張のノードが少し光る

monument:
  相手との未解決トピックが少し立ち上がる

association_graph:
  「無理」「前回の保留」「疲れ」の link ができる

ignition:
  recall_anchor は出るが、confidence はまだ低い

forgetting / tension:
  今は記憶を強く再構成すると危険

memory_dynamics:
  dominant_relation_type: unfinished_carry
  ignition_mode: arming or primed
  memory_tension: medium

closure_packet:
  do_not_overinterpret
  leave_return_point
  low_basis_confidence

reaction_contract:
  interpretation_budget: none
  question_budget: 0
  closure_mode: leave_open
```

悪い返答:

```text
前のことがまだ不安なんだね。
```

これは記憶と相手の内面を決めつけています。

よい返答:

```text
そっか。今は無理に言葉にしなくて大丈夫。
あとで戻れそうなら、そのときでいいよ。
```

ここでは、記憶の気配は使っていますが、記憶の内容は押しつけていません。

## 例: 「この前の話、まだ少し残ってる」

相手:

```text
この前の話、まだ少し残ってる。
```

この場合は、相手自身が過去の話を持ち出しています。

そのため、発火状態は強くなりやすいです。

```text
recall_anchor:
  previous_thread

ignition_mode:
  active

dominant_relation_type:
  unfinished_carry

dominant_causal_type:
  reopened_by
```

ただし、それでも `ClosurePacket` は確認します。

```text
basis_confidence は十分か
memory_tension は高すぎないか
相手が今、説明を求めているのか
ただ残っていることを共有したいだけなのか
```

返答は、たとえばこうです。

```text
うん、まだ残っているんだね。
今ここで全部ほどかなくてもいいから、まず残っている感じだけ一緒に置いておこう。
```

ここでも、発火した記憶をそのまま解釈に使うのではなく、距離と解釈量を調整しています。

## まとめ

`memory_dynamics` は、メモリーパレス、モニュメント、発火状態、再固定化、忘却、連想グラフを1つの typed slow-state にまとめます。

```text
memory_palace
  記憶地図。どの場所が光っているか。

monument
  長く残る意味のアンカー。どの出来事や関係が重いか。

association_graph
  seed 同士のつながり。どの記憶が何と結びつくか。

ignition
  今まさに立ち上がるか。再想起が発火するか。

reconsolidation
  何を再生・固定し直すか。

forgetting
  何を今は使わない方がよいか。

memory_dynamics
  それらを返答制御に使える状態軸へまとめる。

ClosurePacket
  その状態軸を、返答の根拠・制約・不確かさへ変換する。

reaction_contract
  最終的にどう振る舞うかを決める。
```

したがって、EQNet における記憶は、検索対象ではありません。

記憶は、今の返答がどのくらい近づくべきか、どのくらい言葉にすべきか、どの記憶を言わずに効かせるべきかを決める、反応地形です。
