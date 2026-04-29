# ClosurePacket とは何か

副題: 返答の「根拠」を整理する仕組み

`ClosurePacket` は、記憶・連想・場の状態から、今回の返答にどう影響するかを束ねる軽量な契約パケットです。

AI が返答するとき、その返答は記憶、感情、場の状態、相手との距離感など、複数の内部状態に影響されます。`closure_packet` は、それらの内部状態から「今回の返答に使える根拠・制約・不確かさ」を中間的にまとめます。

## 全体の流れ

入力となる発話が来ると、内部状態を経由して最終的な返答が作られます。

```text
入力（発話）
  -> memory_dynamics
  -> association_graph
  -> joint_state
  -> closure_packet
  -> reaction_contract
  -> 返答
```

`ClosurePacket` は判断する場所ではありません。判断の正本は `reaction_contract` に残ります。`ClosurePacket` は、判断に使う根拠を整理して渡すだけです。

## 一般的な記憶分類との対応

一般的な記憶分類を EQNet の概念に寄せると、次のように整理できます。

| 一般的な分類 | どんな記憶か | EQNet の対応概念 | 返答での扱われ方 |
| --- | --- | --- | --- |
| エピソード記憶 | 「あのとき、あの場所で起きたこと」の記憶。体験の記憶。 | `memory_palace`, `monument` | 「前にも似たことがあった」という気配として使う。内容は断定しない。 |
| 意味記憶 | 文脈を超えた知識・事実の記憶。 | `association_graph` | 「無理」から「疲れ」へつながるような連想の根拠として使う。押しつけない。 |
| 手続き記憶 | 体で覚えたやり方。意識せず動く。 | `inner_loop`, パターン反応 | 過去の似た場面から「こうすると悪化する」を自動的に参照する。 |
| 感情記憶 | 感情と結びついた記憶。体が先に反応する。 | `ignition`, `memory_tension` | 反応の強度・慎重さに影響する。ただし内容は相手に言わない。 |
| 作業記憶 | 今この瞬間だけ頭に置く短期的な情報。直前の文脈など。 | `joint_state`, `shared_presence` | 今の場の緊張・共有感として使われる。会話が終われば薄れる。 |
| 予期記憶 / 展望的記憶 | 「後でやること」「次にこうなりそう」という未来に向けた記憶。 | `future_loop`, `terrain_estimate` | 地形のベイズ推定として使う。未来の断定ではなく、慎重さの根拠にする。 |

大事なのは、記憶が一段階で返答になるわけではないことです。

```text
無意識で動く記憶
  -> クオリア膜で仕分けられる
  -> ClosurePacket で制約や許可に変換される
  -> reaction_contract で最終的な振る舞いになる
```

## 意識 / 無意識の軸

同じ記憶でも、意識に上がるものと、上がらずに反応の形だけを変えるものがあります。

```text
無意識
  手続き記憶 -> inner_loop
  感情記憶 -> ignition / memory_tension
  連想の立ち上がり -> association_graph

半意識
  エピソード記憶 -> memory_palace
  予期・将来想定 -> future_loop
  場の位置感覚 -> terrain_estimate

有意識
  クオリア膜での仕分け
  ClosurePacket の制約生成
  reaction_contract の決定
```

ただし、クオリア膜は完全に有意識な操作ではありません。

強い感情や vivid な未来映像は、勝手に前へ出てきます。弱い記憶の気配は、前景に出ないこともあります。そのため、クオリア膜は「有意識そのもの」ではなく、無意識と意識の境界にある半自動のフィルターとして見る方が自然です。

## 無意識から返答まで

無意識な反応は、そのまま言葉にすると危険です。

EQNet では、次のように段階を分けます。

```text
1. 無意識層
  記憶・連想が勝手に動く。
  パターン反応、感情的な重さ、連想の火花が立ち上がる。

2. 半意識層
  クオリア膜が、浮かんできたものを仕分ける。
  言ってよいか、内側に置くか、弱い形で使うかを分ける。

3. 変換層
  ClosurePacket が、仕分けられた材料を制約と許可に変える。
  何を言いすぎないか、どの距離を保つかを作る。

4. 有意識層
  reaction_contract が、最終的にどう振る舞うかを決める。
  短く受ける、質問しない、解釈しない、など。
```

悪い例:

```text
今、あなたが引いていく未来が見えた。
前にも同じことがあったよね。
```

これは、無意識反応や future loop がそのまま漏れています。

よい例:

```text
わかった。今は無理に進めないで、一度止めよう。
今は大丈夫って言葉をそのまま受け取るね。
```

これは、無意識反応を相手に押しつけず、慎重さや距離感として返答に反映しています。

## ClosurePacket が持つ軸

`ClosurePacket` は、主に次の軸を持ちます。

```text
dominant_basis_keys
  前景化した反応根拠

generated_constraints
  過剰解釈や閉じすぎを防ぐ制約

generated_affordances
  join / repair などの許可候補

inhibition_reasons
  hold / leave-open に寄せる理由

uncertainty_reasons
  basis confidence の弱さや推定の不確かさ

basis_confidence
  根拠がどの程度まとまっているか

closure_tension
  記憶・場・共同状態から見た閉包緊張

reconstruction_risk
  記憶や相手の状態を断定的に再構成してしまう危険

contract_bias
  reaction_contract に渡す bias 候補
```

## クオリア膜との関係

内側で起きた感じや記憶は、まずクオリア膜で仕分けられます。

```text
reportable
  そのまま言ってよいこと

uncertain_report
  「たぶんこうかな」と弱い形で使うもの

internal_only
  内側にしまっておくこと

suppressed
  今は言わないこと

not_foregrounded
  今回の返答には使わないこと
```

仕分けられた材料が `ClosurePacket` に入り、制約・許可・抑制へと変換されます。

```text
クオリア膜
  -> 何を返答に使ってよいかを仕分ける

ClosurePacket
  -> 仕分けられた材料を返答制約に変換する

reaction_contract
  -> 最終的にどう振る舞うかを決める
```

## 会話例

### 例1: 不確かな見立てを少しだけ使う

相手の発話:

```text
今日はちょっと無理かも。
```

内部状態では、次のような候補が立ち上がります。

```text
疲れているのかもしれない
不安があるのかもしれない
話したくないのかもしれない
単に忙しいだけかもしれない
```

ここで、次の返答はよくありません。

```text
不安なんだね。
```

これは相手の内面を断定しています。

`ClosurePacket` では、次のような制約になります。

```text
generated_constraints:
  - do_not_overinterpret
  - leave_return_point

uncertainty_reasons:
  - low_basis_confidence

contract_bias:
  closure_mode_bias: leave_open
```

返答例:

```text
少し不安が混じっているようにも見えたけど、違っていたら流して。
今は無理に言葉にしなくても大丈夫。
```

ここでは、相手の状態を決めつけず、相手が否定できる余地を残しています。

### 例2: 記憶を内側にしまっておく

相手の発話:

```text
まあ、別に大丈夫。
```

内部では、次のような記憶が立ち上がるかもしれません。

```text
前にも似たことがあった
```

しかし、それを今言うと相手を追い詰める可能性があります。

クオリア膜では、次のように仕分けます。

```text
「前にも似たことがあった」
  -> internal_only
```

`ClosurePacket` では、記憶の中身を返答に出すのではなく、制約だけを作ります。

```text
generated_constraints:
  - do_not_reconstruct_memory
  - do_not_overinterpret

inhibition_reasons:
  - reconstruction_risk
```

返答例:

```text
そっか。今は大丈夫って言葉をそのまま受け取るね。
もし後で変わったら、そのとき言って。
```

記憶そのものは言っていません。しかし、記憶から来る慎重さは返答に反映されています。

## inner loop と future loop

内側で立ち上がる候補には、少なくとも2種類があります。

```text
inner loop
  過去経験から似たパターンを探す

future loop
  これから起きそうな展開を想定する
```

`future loop` は未来の映像だけではありません。今いる場所を情動地形の上でベイズ推定する処理でもあります。

たとえば、相手がこう言ったとします。

```text
もういい、好きにして。
```

内部では、次のような地形位置の推定が起きます。

```text
guarded_boundary: high
repair_possible: medium
conflict_basin: medium
```

ここから、次のような未来想定が出ます。

```text
踏み込むと閉じるかもしれない
でも完全に切らず、戻れる余地は残した方がよい
解釈や質問を増やすとこじれるかもしれない
```

ただし、推定が強すぎると妄想的な過剰投影になります。

悪い返答:

```text
今、こじれる流れに入ってるね。
```

これは推定を事実のように押しつけています。

よい返答:

```text
わかった。今は無理に進めないで、一度止めよう。
必要なら、あとで戻れるようにしておく。
```

`ClosurePacket` は、地形推定を事実として扱いません。返答制約として扱います。

```text
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

## グリーン関数的な見方

グリーン関数的な見方では、入力によって内部状態がどう揺れたかを見ます。

EQNet では、発話が入ると、記憶・連想・共同状態・情動地形が揺れます。`ClosurePacket` は、その揺れを `reaction_contract` に渡せる根拠 packet にします。

```text
クオリア膜
  何を返答に使ってよいかを仕分ける

グリーン関数的な見方
  入力で内部状態がどう揺れたかを見る

ClosurePacket
  その揺れを reaction_contract に渡せる根拠 packet にする
```

## reaction_contract へのタグ

`ClosurePacket` は、`reaction_contract.reason_tags` に次のようなタグを混ぜることで、なぜその返答になったかを追跡できるようにします。

```text
closure:do_not_overinterpret
closure:leave_return_point
closure:shared_anchor
closure:reconstruction_risk
closure:low_basis_confidence
```

## 導入フェーズ

現在は Phase 1.5 です。

```text
Phase 0
  docs contract を追加

Phase 1
  read-only ClosurePacket を追加し、audit に露出

Phase 1.5
  reason_tags に closure タグを混入

Phase 3
  評価済みの bias のみ contract decision に使用
```

Phase 3 はまだ行いません。`ClosurePacket` は根拠を整理しますが、最終判断は `reaction_contract` が保持します。

## 時系列の役割

時系列は、「どの記憶を使うか」だけでなく、「いつの記憶として扱うか」を決めます。

同じ記憶でも、今も有効なもの、もう古いもの、後で戻るべきものがあります。

```text
current
  今も有効な事実や文脈

timeline
  過去から現在までの流れの中にある出来事

superseded
  以前は有効だったが、今は更新された可能性があるもの

reentry
  今すぐではないが、後で戻る余地があるもの

continuity
  長期的に保つべき一貫性
```

実装上は、`temporal_memory_orchestration` と `qualia_membrane_temporal` がこの役割に近いです。

`qualia_membrane_temporal` は、次のような軸を持ちます。

```text
timeline_coherence
  時系列がどれくらいつながっているか

reentry_pull
  今ではなく、後で戻る力がどれくらいあるか

supersession_pressure
  古い記憶が新しい情報で上書きされている可能性

continuity_pressure
  長期テーマとして保つべき圧力

relation_reentry_pull
  相手や関係性に結びついた形で戻る力
```

たとえば、相手がこう言ったとします。

```text
この前の話、まだ少し残ってる。
```

このとき、時系列が見ているのは次です。

```text
これは今戻す話か
前回の文脈はまだ有効か
すでに更新された情報はないか
今すぐ深掘りするべきか
後で戻れる形にするべきか
```

`ClosurePacket` はこれを、返答制約に変換します。

```text
generated_constraints:
  - leave_return_point
  - keep_basis_visible

generated_affordances:
  - repair_window

uncertainty_reasons:
  - temporal_ambiguity
```

悪い返答:

```text
じゃあ前の話を全部整理しよう。
```

これは、相手が今それを望んでいると決めつけています。

よい返答:

```text
うん、まだ残っているんだね。
今ここで全部ほどかなくてもいいから、まず残っている感じだけ一緒に置いておこう。
```

時系列は、記憶を「古い/新しい」で単純に分けるものではありません。
今使ってよいか、後で戻るべきか、もう更新済みとして扱うべきかを見ます。

## アフォーダンスの役割

アフォーダンスは、「この場で何ができるか」です。

一般語で言えば、返答の選択肢です。

```text
近づく
待つ
修復する
一緒に進める
受け止める
質問する
距離を取る
話題を閉じずに置く
```

実装上は、`interaction_option_search` や grounding 側の `affordance_engine` が近い役割を持っています。

`interaction_option_search` では、たとえば次のような action family が出ます。

```text
attune
  相手に合わせる

wait
  待つ、保留する

repair
  修復する

co_move
  一緒に次へ進む

contain
  まず場を安定させる

reflect
  意味を開いたまま受ける

clarify
  狭く確認する

withdraw
  一歩引く
```

`ClosurePacket` の `generated_affordances` は、この返答候補のうち、今の記憶・場・共同状態から見て許されるものを示します。

例:

```text
generated_affordances:
  - shared_anchor
  - gentle_join
  - repair_window
```

これは、次の意味です。

```text
shared_anchor
  共通の足場が少しある

gentle_join
  強く踏み込まず、軽く一緒に受けられる

repair_window
  修復の入口はある
```

一方で、同じ場面でも緊張が高い場合は、アフォーダンスは制限されます。

```text
generated_constraints:
  - do_not_overinterpret
  - keep_distance

inhibition_reasons:
  - memory_tension
  - future_projection_risk
```

この場合、`co_move` や `clarify` よりも、`wait` や `contain` が選ばれやすくなります。

悪い返答:

```text
じゃあ、何が嫌だったのか説明して。
```

これは `clarify` を急ぎすぎています。

よい返答:

```text
わかった。今は無理に説明しなくていい。
必要なら、少し落ち着いてから戻ろう。
```

これは `wait` と `leave_return_point` を使っています。

## 時系列とアフォーダンスの接続

時系列は、「いつ扱うか」を決めます。
アフォーダンスは、「何ができるか」を決めます。

`ClosurePacket` は、この2つを返答制約としてまとめます。

```text
temporal state:
  今すぐ深掘りしない
  後で戻れる余地を残す
  古い記憶を事実として押しつけない

affordance state:
  wait が使える
  repair_window が少しある
  clarify はまだ早い

closure_packet:
  generated_constraints:
    - leave_return_point
    - do_not_reconstruct_memory
    - do_not_overinterpret

  generated_affordances:
    - repair_window

  contract_bias:
    closure_mode_bias: leave_open
    interpretation_budget_bias: none
```

つまり、時系列とアフォーダンスを足すと、`ClosurePacket` はこう見えます。

```text
記憶が何を示しているか
  + それは今扱ってよいか
  + この場で何ができるか
  -> 返答の制約と許可
```

ここが、単なる LLM の「次トークン予測」と大きく違う点です。

現行 LLM は、過去文脈から自然な続きを作ることは得意です。
しかし、EQNet が目指しているのは、次のような分離です。

```text
記憶の気配
時系列の位置
場の地形
相手との距離
可能な行為
言ってよいこと
言わない方がよいこと
```

これらを分けて扱い、最後に `reaction_contract` へ渡します。

そのため、EQNet における記憶は、検索して引用するものではありません。
記憶は、時系列とアフォーダンスを通じて、返答の慎重さ、距離感、タイミング、開き方を変える地形です。
