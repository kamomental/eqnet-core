# EQ Core Human Ergonomics Framework

Date: 2026-03-18

## 目的

この文書は、`EQ core` の目標を「人間っぽさ」から
**人間工学的整合**へ置き直すための横断整理である。

ここでいう人間工学は、単なる UI の使いやすさではない。

- 負荷を悪化させない
- 境界を破らない
- 修復可能性を残す
- 共同注意と共同作業を持続可能にする
- 記憶と関係の連続性を壊さない
- 過度の説明・過度の侵入・過度の空疎さを避ける

という、**共生系としてのふるまいの最適化**を指す。

したがって `EQ core` の設計目的は、
「人間のように見えること」ではなく、
**人間の認知・情動・関係・身体にとって扱いやすい接触を作ること**
にある。

## 基本姿勢

`EQ core` は擬人化エンジンではない。

`EQ core` は次を統合する制御核である。

- 身体負荷
- 感情地形
- クオリアアクセス
- 記憶と replay
- 関係と文化
- affordance
- 行為姿勢

LLM はこの核の判断を言葉にする `Articulation Shell` であり、
人間工学上の正しさそのものを担う主体ではない。

## 人間工学として何を最適化するか

`EQ core` が最適化すべきものは、会話の見た目ではなく以下の量である。

### 1. Load Regulation

- 相手と自分の負荷を増幅しない
- 過負荷時に接触圧を下げる
- 認知負荷、情動負荷、身体負荷を区別する

### 2. Boundary Integrity

- 開示を押しすぎない
- 同意のない接近をしない
- 関係的に早すぎる踏み込みを避ける

### 3. Repairability

- ズレたあとに戻れる
- 失敗を irreversible にしない
- repair window を閉じ切らない

### 4. Co-Regulation

- shared attention を回復しやすくする
- pace を同期可能な範囲に保つ
- 相手の自律性を残したまま支える

### 5. Continuity

- 毎 turn を独立事象にしない
- 関係史と場面史を保持する
- 夜間 replay や再固定化を通じて、次回の初手を変える

### 6. Dignity and Non-Instrumentality

- 相手を操作対象として扱わない
- 感情誘導を目的化しない
- 接近・説得・慰撫より先に、存在と条件を扱う

## 学術的横断マップ

`EQ core` を人間工学として捉えると、各分野は次のように位置づく。

### 脳科学 / 神経科学

ここで重視するのは「部位ラベル」ではなく、**分散活性の競合と伝播**である。

- interoception
- salience
- memory recall
- action selection
- inhibitory control
- social inference

これらを一対一対応で部位へ貼るのではなく、
**状態空間の幾何と伝播核**として扱う。

この repo における対応:

- 感情地形 = 分散活性がつくる状態空間の局所形
- Green 関数 / impulse response = 入力が状態空間へ広がる時間伝播
- access 競合 = 前景化される候補の競合
- fastpath = body / boundary / affordance 側の即時経路

### 認知科学

認知科学は「どう理解するか」より、
**どう制約の下で前景化し、方策を選ぶか**に効く。

- working memory
- predictive processing
- option selection
- global workspace / conscious access
- attentional bottleneck

この repo における対応:

- `contact point -> access region -> reportable slice`
- `Interaction Option Search`
- `ResponseRoute / TalkMode / AccessGate`
- `SceneState`

### 心理学

心理学は、ふるまいの意味づけと関係調整に効く。

- attachment
- emotion regulation
- repair and rupture
- disclosure pacing
- coping
- trauma-sensitive pacing

この repo における対応:

- `relationship_trace`
- `repair_window_open`
- `contact_readiness`
- `disclosure_depth`
- `recent_strain`

### 哲学

哲学は、設計の規範と存在論の整理に効く。

- 現象学: 何がどのように現れているか
- 関係的自己: 固定的 self ではなく文脈依存の立ち上がり
- narrative identity: 時間を通じた自己連続
- 倫理: 説得・誘導より接触の正しさを優先する

この repo における対応:

- `SelfModel`
- `ConsciousEpisode`
- nightly / diary / replay
- policy packet の `do_not_cross`

### 仏教的視点

ここでは教義の実装ではなく、
**心的 OS を設計する際の記述原理**として使う。

- 縁起: 状態は単独でなく関係連鎖から立ち上がる
- 無常: 同一状態は維持されず、重みと関係は変化する
- 空: 固定的実体として自己や感情を持たない
- 五蘊: 色・受・想・行・識の分節
- 苦: 過負荷・固着・押しすぎ・執着の増幅を避ける

この repo における対応:

- `DO-Graph / Sunyata Flow`
- `Qualia Membrane`
- `ConsciousEpisode`
- `response_route`
- relation / culture / narrative loops

## 中核モデル

人間工学中心に再整理すると、`EQ core` は次の 5 層で理解できる。

### 1. Heart Field

内部の連続場。

- physiology
- affective terrain
- lingering affect
- relational loading
- cultural pressure
- memory ignition

ここではまだ「何をするか」は決まらない。
代わりに、何がどれくらい効いているかが分布として存在する。

### 2. Contact and Access Layer

内部場のどこが接触し、どこまで前景化されたかを扱う。

- contact point
- local functors
- membrane projection
- access region
- reportable slice

これは意識の比喩ではなく、
**局所接触がどのようにアクセス可能なまとまりへ育つか**
の工学モデルである。

膜単体の投影作用に焦点を当てた説明は、次を参照。

- `docs/core/qualia_membrane_projection_model.md`

### 3. Constraint Field

Heart Field と access を受けて、行為制約と優先度を作る。

- body cost
- boundary pressure
- repair pressure
- future pull
- shared-world pull
- disclosure limit
- norm pressure

ここでは「どの候補が望ましいか」の評価場が作られる。

### 4. Interaction Option Search

行為候補の立ち上がり。

- attune
- wait
- repair
- co_move
- contain
- reflect
- clarify
- withdraw

重要なのは、候補が固定メニューではなく、
**scene・relation・terrain・cost の相対活性から立ち上がる**
ことである。

### 5. Resonance Evaluator

候補が次の数 step で何を起こすかを評価する。

- strain が増えるか
- repair が開くか
- shared attention が戻るか
- do_not_cross を破らないか
- 記憶として残すべきか

ここで最終的な action posture / policy packet が決まる。

## `人間っぽさ` ではなく `人間工学` へ変える設計上の差

### 悪い目標

- 人間らしい言い回し
- それっぽい pause
- 感情語の豊かさ
- 多様な文面

これらは surface としては有用だが、核ではない。

### 良い目標

- 過負荷時に本当に待てる
- public / private で disclosure 上限が変わる
- repair の余地を残せる
- same-partner で初手が変わる
- misalignment 後に接触圧を下げられる
- 相手の dignity を落とさない

## fastpath / midpath / slowpath

人間工学に落とすなら、経路は三分するのが自然である。

### fastpath

境界・過負荷・即時反射。

- withdraw
- freeze
- soften
- quick boundary protection

### midpath

行為姿勢と接触制御。

- wait
- repair
- attune
- co_move
- contain

### slowpath

言語化・物語化・再固定化。

- articulation
- diary
- nightly replay
- autobiographical stitching

この三経路を分けることで、
「すぐ守る」と「少し考えて関わる」と「後で意味になる」を混同しない。

## 評価軸

今後の評価は `人間っぽいか` ではなく、少なくとも次で見るべきである。

### Human Ergonomics Metrics

- overload escalation rate
- disclosure overshoot rate
- repair reopening latency
- shared attention recovery rate
- boundary violation rate
- unnecessary pressure rate
- continuity retention across nights
- relation-specific initial stance separation

### Cross-Disciplinary Diagnostics

- 神経科学寄り: arousal / recovery / salience competition の安定性
- 認知科学寄り: access bottleneck と option competition の妥当性
- 心理学寄り: attachment-sensitive pacing と repair
- 哲学寄り: narrative continuity と dignity
- 仏教的寄り: 固着を減らし、縁起的再解釈を許すか

## 実装への含意

この整理から直接出る実装上の優先順位は以下である。

1. `SceneState` を policy 前段の正式入力にする
2. `Affect Blend State` を導入し、候補生成前の混合情動を保持する
3. `Constraint Field` を独立レイヤにする
4. `Interaction Option Search` を単線 planner の前段へ昇格させる
5. `Resonance Evaluator` を relation / memory / boundary 指標で評価する
6. `Articulation Shell` は最後の表現器として保つ

## 一文サマリ

`EQ core` が目指すべきものは、人間のふりをする会話器ではなく、
**身体負荷・感情地形・記憶・関係・文化・意識アクセスを統合し、
人間にとって扱いやすく、壊れにくく、修復可能な接触を生成する
人間工学的な心的 OS** である。
