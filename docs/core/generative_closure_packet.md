# Generative Closure Packet

## 目的

`closure_packet` は、新しい巨大な生成閉包層ではなく、`memory_dynamics / association_graph / joint_state / reaction_contract` の間に置く軽量な typed contract です。

役割は、経験・記憶・連想・場の状態から、今回の反応契約に影響しうる根拠、抑制、未確定性、bias を read-only packet として束ねることです。

## 非目的

- 文章を生成しない
- LLM に raw observation を渡さない
- `reaction_contract` の最終判断を置き換えない
- `memory_dynamics` と `joint_state` の責務を吸収しない

`closure_packet` は根拠です。判断の正本は `reaction_contract` のままです。

## 入力

最小入力は次です。

- `memory_dynamics_state`
- `association_graph_state`
- `terrain_dynamics_state`
- `subjective_scene_state`
- `self_other_attribution_state`
- `shared_presence_state`
- `joint_state`

現在の quickstart では、既存 scenario の `memory_dynamics_state` と、core loop から導出済みの `subjective_scene / self_other_attribution / shared_presence / joint_state` から packet を作ります。

## 出力

`ClosurePacket` は次の軸を持ちます。

- `dominant_basis_keys`: 今回の反応根拠として前景化した basis
- `generated_constraints`: 過剰解釈、過剰再構成、閉じすぎを抑える制約
- `generated_affordances`: join、repair、shared anchor などの許可候補
- `inhibition_reasons`: hold / guarded / leave-open に寄せる理由
- `uncertainty_reasons`: basis confidence や association margin の弱さ
- `basis_confidence`: basis のまとまり
- `closure_tension`: 記憶・場・共同状態から見た閉包緊張
- `reconstruction_risk`: 記憶を断定的に再構成する危険
- `contract_bias`: `reaction_contract` に渡せる bias 候補

## reaction_contract との関係

`closure_packet` は `reaction_contract` を決定しません。

現段階では、packet は quickstart の audit に出力され、`reaction_contract.reason_tags` に `closure:*` タグとして混ぜられます。これにより、なぜ `hold / join / interpretation-none / leave-open` に寄ったのかを追跡できます。

例:

- `closure:do_not_overinterpret`
- `closure:leave_return_point`
- `closure:shared_anchor`
- `closure:reconstruction_risk`
- `closure:low_basis_confidence`

## memory_dynamics との関係

`memory_dynamics` は、記憶そのものではなく、連想・記念碑・再想起・再固定・忘却圧を束ねる slow-state です。

`closure_packet` はその下流で、`memory_dynamics` から見えた relation / causal / tension を、反応契約に使える根拠 packet に投影します。

つまり、

`memory_dynamics = 記憶の動力学状態`

`closure_packet = 反応契約へ渡す生成根拠の中間投影`

です。

## 評価項目

評価では、少なくとも次を確認します。

- `do_not_overinterpret` が出た場面で、解釈予算が過剰になっていないか
- `reconstruction_risk` が高い場面で、記憶を断定していないか
- `leave_return_point` が出た場面で、会話を閉じすぎていないか
- `shared_anchor` が出た場面で、自然な join に接続できるか
- `basis_confidence` が低い場面で、initiative を取りすぎていないか

実装入口は `inner_os/evaluation/closure_packet_eval.py` です。既存の `conversation_contract_eval` を置き換えず、closure 専用の検査 hook として併用します。

## 導入段階

現在の導入段階は Phase 1.5 です。

- Phase 0: docs contract を追加
- Phase 1: read-only `ClosurePacket` を追加し、quickstart / audit に露出
- Phase 2: `reaction_contract.reason_tags` に closure 由来タグを混ぜる
- Phase 3: 評価を通った bias だけを contract decision に使う

Phase 3 はまだ行いません。`contract_bias` は観測可能な候補として残し、最終判断は `reaction_contract` が保持します。

## グリーン関数的な見方

グリーン関数は、ある場所に小さな入力を入れたとき、系がどのように応答するかを見るための考え方です。

EQNet では、入力は発話、表情、記憶の立ち上がり、違和感、再想起 seed などです。

例:

```text
入力:
今日はちょっと無理かも。
```

この入力に対して、内部では次のような応答が起きます。

```text
memory_dynamics:
  前にも似た重さがあったかもしれない
  ただし記憶として断定するには弱い

association_graph:
  「無理」「重い」「前回の保留」のつながりが少し立ち上がる

joint_state:
  shared_tension が上がる
  common_ground はまだ低い

closure_packet:
  do_not_overinterpret
  leave_return_point
  reconstruction_risk
```

つまり `closure_packet` は、入力に対する内部状態の応答を、返答の制約として見える形にしたものです。

式のように書くなら、次のように見なせます。

```text
input impulse
  -> memory_dynamics / association_graph / joint_state
  -> closure_packet
  -> reaction_contract
```

ただし、これは普通の線形なグリーン関数ではありません。同じ入力でも、直前の記憶、相手との距離、場の緊張、境界の安定度によって応答が変わります。

そのため、`closure_packet` は次のようなものです。

```text
会話における非線形なインパルス応答の要約
```

## クオリア膜との関係

クオリア膜は、内側で起きた感じや記憶を、そのまま外に出さずに仕分ける場所です。

一般向けには、次の5分類で説明します。

```text
相手にそのまま言ってよいこと（reportable）
内側にしまっておくこと（internal_only）
たぶんこうかな、と思ったことを、決めつけずに言うこと（uncertain_report）
今は言わないこと（suppressed）
今回の返答には使わないこと（not_foregrounded）
```

設計上の役割は次です。

```text
qualia membrane:
  内側で起きた感じや記憶を、返答に使うかどうか仕分ける

closure_packet:
  仕分けられた材料から、返答の制約・許可・抑制・不確かさを作る

reaction_contract:
  最終的に、話す / 黙る / 短く返す / 質問しない / 解釈しない、を決める
```

流れは次です。

```text
raw observation / memory trace / association
  -> qualia membrane
  -> reportable / internal_only / uncertain_report / suppressed / not_foregrounded
  -> closure_packet
  -> reaction_contract
  -> expression bridge
```

## 会話例

### 例1: 不確かな見立てを少しだけ使う

相手:

```text
今日はちょっと無理かも。
```

内側では、いくつかの候補が立ち上がります。

```text
疲れているのかもしれない
不安があるのかもしれない
話したくないのかもしれない
単に忙しいだけかもしれない
```

ここで、決めつける返答はよくありません。

```text
不安なんだね。
```

これは相手の内面を断定しています。

クオリア膜では、これは次のように扱います。

```text
「相手は不安かもしれない」
  -> uncertain_report
```

つまり、確定した事実としては扱いません。

`closure_packet` は、この不確かさを次のような制約へ変換します。

```text
generated_constraints:
  - do_not_overinterpret
  - leave_return_point

uncertainty_reasons:
  - low_basis_confidence

contract_bias:
  interpretation_budget_bias: none
  closure_mode_bias: leave_open
```

返答は、たとえばこうなります。

```text
少し不安が混じっているようにも見えたけど、違っていたら流して。
今は無理に言葉にしなくても大丈夫。
```

ここで重要なのは、次の3点です。

```text
「あなたは不安だ」と決めつけない
「そう見えた」という弱い形にする
「違っていたら流して」と相手が否定できる余地を残す
```

### 例2: 内側にしまっておく

相手:

```text
まあ、別に大丈夫。
```

内側では、過去の似た場面が立ち上がるかもしれません。

```text
前にも「大丈夫」と言いながら無理していたことがあった
```

ただし、今それを言うと、相手を追い詰める可能性があります。

クオリア膜では、こう仕分けます。

```text
「前にも似たことがあった」
  -> internal_only
```

`closure_packet` は、記憶の中身を返答に出すのではなく、制約だけを作ります。

```text
generated_constraints:
  - do_not_reconstruct_memory
  - do_not_overinterpret

inhibition_reasons:
  - reconstruction_risk
```

返答は、たとえばこうです。

```text
そっか。今は大丈夫って言葉をそのまま受け取るね。
もし後で変わったら、そのとき言って。
```

記憶そのものは言っていません。しかし、記憶から来る慎重さは返答に反映されています。

### 例3: 今回の返答には使わない

相手:

```text
この話、前にもしたっけ。
```

内部には関連する記憶がいくつかあります。

```text
似た話題
似た感情
似た相手の反応
```

しかし、今回の返答では、細かい記憶を掘り返す必要がない場合があります。

クオリア膜では、こう仕分けます。

```text
「関連しそうな記憶」
  -> not_foregrounded
```

`closure_packet` は、記憶を話題の中心にしない制約を作ります。

```text
generated_constraints:
  - keep_basis_visible
  - do_not_reconstruct_memory
```

返答は、たとえばこうです。

```text
似た話はあった気がするけど、今は細かく掘らずに、この話として聞くね。
```

## inner loop と future loop

内側で立ち上がる候補には、少なくとも2種類があります。

```text
inner loop:
  過去経験や記憶断片から、今の状況に似たパターンを探す

future loop:
  このまま進むと何が起きそうかを、先回りして想定する
```

ただし、future loop は未来の映像だけではありません。同時に、今いる場所を情動地形の上で推定する処理でもあります。

言い換えると、次の2つは同じ現象の別の見方です。

```text
future loop:
  このまま進むと、次に何が起きそうかを想定する

terrain position Bayesian estimation:
  今の会話が、どの地形位置にいるのかを確率的に推定する
```

たとえば、相手の発話が次だったとします。

```text
もういい、好きにして。
```

このとき、内部では未来だけでなく、現在位置の推定も起きます。

```text
地形候補:
  repair_possible: まだ修復できる位置かもしれない
  guarded_boundary: 境界が固くなっている位置かもしれない
  withdrawal_slope: 引いていく斜面に入っているかもしれない
  conflict_basin: こじれやすい谷に近いかもしれない
```

これは1つに決め打ちしません。複数の仮説に確率を置きます。

```text
P(guarded_boundary | current_signals) = high
P(repair_possible | current_signals) = medium
P(conflict_basin | current_signals) = medium
P(open_play | current_signals) = low
```

この推定があるから、future loop は次のような予測を出します。

```text
guarded_boundary が高い
  -> 踏み込むと閉じるかもしれない

repair_possible も残っている
  -> 完全に切らず、返り道を残す方がよい

conflict_basin が中くらいある
  -> 解釈や質問を増やすとこじれるかもしれない
```

つまり、future loop は単なる空想ではなく、地形位置のベイズ推定から出る予測です。

ただし、ここにも偏りのリスクがあります。

```text
過去の失敗が強すぎる
  -> P(conflict_basin) を過大評価する

拒絶への感度が高すぎる
  -> P(withdrawal_slope) を過大評価する

相手との共有地が見えていない
  -> P(repair_possible) を過小評価する
```

この偏りが強いと、未来想定は妄想的な過剰投影になります。

そのため、`closure_packet` は地形推定を事実として扱いません。返答制約として扱います。

```text
terrain estimate:
  guarded_boundary: high
  repair_possible: medium
  conflict_basin: medium

closure_packet:
  generated_constraints:
    - do_not_overinterpret
    - leave_return_point
    - keep_distance

  inhibition_reasons:
    - terrain_position_uncertainty
    - future_projection_risk

  contract_bias:
    interpretation_budget_bias: none
    distance_mode_bias: guarded
    closure_mode_bias: leave_open
```

悪い返答:

```text
今、こじれる流れに入ってるね。
```

これは推定を事実のように言っています。

よい返答:

```text
わかった。今は無理に進めないで、一度止めよう。
必要なら、あとで戻れるようにしておく。
```

ここでは、地形推定を相手に押しつけていません。しかし、推定から来た慎重さは反応に反映されています。

たとえば、相手がこう言ったとします。

```text
もういい、好きにして。
```

このとき、inner loop は過去の似た場面を探します。

```text
前にも似た言い方があった
そのとき踏み込んで悪化した
別のときは、一度止めたら修復できた
```

一方で、次のようなものは future loop です。

```text
相手が引いていく映像
会話が硬くなる場面
その場の空気が狭くなるイメージ
このまま質問すると、さらに閉じそうな予感
```

これは過去そのものではなく、過去経験を材料にした未来想定です。

future loop は有用です。危ない踏み込みを避けたり、質問を減らしたり、短く受けたりする助けになります。

ただし、偏りが強いと危険です。

```text
未来想定が強すぎる
  -> 相手がまだ何もしていないのに、拒絶されると決めつける

過去の失敗が強く残りすぎる
  -> 今の相手ではなく、過去の場面に反応してしまう

映像や予感が vivid すぎる
  -> 予測ではなく事実のように感じてしまう
```

この状態では、future loop は妄想的な過剰投影になります。

そのため、クオリア膜は future loop をそのまま返答に使わせません。

```text
「相手が引いていく映像が出た」
  -> internal_only

「このまま踏み込むと悪化しそう」
  -> uncertain_report または suppressed

「質問を減らす方がよさそう」
  -> reaction constraint として使う
```

`closure_packet` は、future loop の中身を相手に説明するのではなく、返答制約に変換します。

```text
generated_constraints:
  - do_not_overinterpret
  - leave_return_point
  - keep_distance

inhibition_reasons:
  - future_projection_risk
  - pattern_match_risk

uncertainty_reasons:
  - low_basis_confidence

contract_bias:
  interpretation_budget_bias: none
  distance_mode_bias: guarded
```

悪い返答:

```text
今、あなたが引いていく未来が見えた。
```

これは相手には重く、予測を事実のように押しつけています。

よい返答:

```text
わかった。今は無理に進めないで、一度止めよう。
```

この返答では、future loop の映像は言っていません。しかし、その映像から来た慎重さは反応制約として効いています。

短くまとめると、次です。

```text
inner loop:
  過去経験から似たパターンを探す

future loop:
  これから起きそうな展開を想定する

qualia membrane:
  その想定を、言ってよいもの / 内側に置くもの / 不確かさつきで使うものに分ける

closure_packet:
  未来想定を、断定ではなく返答制約に変換する

reaction_contract:
  実際に、短く受ける / 質問しない / 距離を取る / 解釈しない、を決める
```

## 3つの関係のまとめ

```text
クオリア膜:
  内側で起きたものを、返答に使ってよい形に仕分ける

グリーン関数的な見方:
  入力に対して、記憶・連想・共同状態がどう揺れたかを見る

ClosurePacket:
  その揺れを、返答の制約・許可・抑制・不確かさとしてまとめる
```

短く言うと、次です。

```text
クオリア膜は、何を返答に使ってよいかを仕分ける。
グリーン関数的な見方は、入力で内部状態がどう揺れたかを見る。
ClosurePacket は、その揺れを reaction_contract に渡せる根拠 packet にする。
```
