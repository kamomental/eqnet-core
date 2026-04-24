# Evaluation Targets

## 関連文書
- `docs/core/evaluation_criteria.md`
- `docs/core/evaluation_snapshot_2026_03_28.md`
- `docs/core/evaluation_operating_policy.md`

## 目的

`docs/core/evaluation_criteria.md` と
`docs/core/evaluation_snapshot_2026_03_28.md`
を、開発優先順位に使える **数値目標** に落とす。

ここでの数値は、2026-03-28 時点の repo 内評価に基づく
内部運用用スコアであり、外部公開ベンチのスコアではない。

---

## スコアの意味

- 0: 未着手
- 25: 機序だけある
- 50: 局所では成立
- 75: 一般利用にかなり近い
- 90: 外部評価へ出せる完成度
- 100: この repo の現段階で狙う上限

---

## 現在地と目標値

| 評価軸 | 現在値 | 次の目標 | 中期目標 | 判定メモ |
|---|---:|---:|---:|---|
| 一般会話自然さ | 68 | 78 | 86 | generic fallback は減ったが、一般層向けの語感はまだ改善余地がある |
| 深い話の受け止め | 82 | 88 | 92 | 内容反映は立つ。継続ターンでの揺れをさらに自然化したい |
| 数ターン継続 | 64 | 76 | 85 | thread はあるが、live の 2〜5 ターンで滑らかさがまだ不足 |
| キャラクタ / 存在感 | 66 | 77 | 86 | 機序は立っているが、一般層が読むとまだ少し「支える説明」が見える |
| 機序の実在性 | 88 | 92 | 95 | Green / contact / residual / boundary はかなり成立している |
| 総合到達度 | 73 | 81 | 89 | 内部は強い。自然な表出と multi-turn で仕上げる段階 |

---

## 各軸の到達条件

### 1. 一般会話自然さ

現在値: `68`
次の目標: `78`

`78` に上げる条件:

- 1 文目で何を受け取ったかが分かる
- `もちろんです / どこから続けますか` 系が主要ケースで出ない
- `大丈夫です` の反復がさらに減る
- short final が説明文より会話の地の文に寄る

次に効く実装:

- deep disclosure と thread reopening の compact phrase をさらに分ける
- generic support の route / compaction 混入を減らす

### 2. 深い話の受け止め

現在値: `82`
次の目標: `88`

`88` に上げる条件:

- 1 ターン目で内容反映が安定する
- guarded な場面では reflection-only で止まれる
- 2〜4 ターン目でも generic support に崩れない

次に効く実装:

- multi-turn deep talk fixture を新設
- `reflect_only / light question / quiet presence` の揺れを持たせる

### 3. 数ターン継続

現在値: `64`
次の目標: `76`

`76` に上げる条件:

- thread anchor が論点核として保たれる
- 3〜5 ターンで reopening が generic continuation に戻らない
- residual が caution / hesitation / reopening に再出現する

次に効く実装:

- `discussion_thread_registry` の anchor 抽出を論点核ベースに寄せる
- residual を thread 側へさらに強く返す

### 4. キャラクタ / 存在感

現在値: `66`
次の目標: `77`

`77` に上げる条件:

- content / protection / reopening の癖が turn をまたいで見える
- 一般層が読んで「相談窓口」より「その場で返している感じ」がある
- role / distance / warmth のぶれが減る

次に効く実装:

- human-facing surface fixture を追加
- deep talk と一般会話の phrasebank をもう一段分ける

### 5. 機序の実在性

現在値: `88`
次の目標: `92`

`92` に上げる条件:

- `GreenKernelComposition` の observability が same-turn summary に流れる
- `memory / affective / relational` の主導度を観測できる
- `contact / boundary / residual` の役割分離がさらに明示される

次に効く実装:

- continuity summary に green composition 主導度を追加
- dashboard に kernel composition の概観を出す

---

## 優先順位

次の実装順は、スコア改善効率で決める。

1. 数ターン継続 `64 -> 76`
2. 一般会話自然さ `68 -> 78`
3. キャラクタ / 存在感 `66 -> 77`
4. 深い話の受け止め `82 -> 88`
5. 機序の実在性 `88 -> 92`

理由:

- いま最も不足しているのは multi-turn での滑らかさ
- 一般層に価値として届くには、そこが最優先

---

## 進捗の見方

### 直近の合格ライン

- 一般会話自然さ `>= 78`
- 深い話の受け止め `>= 88`
- 数ターン継続 `>= 76`
- キャラクタ / 存在感 `>= 77`
- 機序の実在性 `>= 92`
- 総合到達度 `>= 81`

このラインを超えたら、
`一般層に価値が伝わり始める段階`
と判定する。

### 中期の合格ライン

- 一般会話自然さ `>= 86`
- 深い話の受け止め `>= 92`
- 数ターン継続 `>= 85`
- キャラクタ / 存在感 `>= 86`
- 機序の実在性 `>= 95`
- 総合到達度 `>= 89`

このラインを超えたら、
`外部ベンチへ本格的に載せる段階`
と判定する。

---

## 改善ループ運用ルール

### ベンチ最適化を避ける原則

- ベンチだけ上がる局所調整は採用しない
- live の自然さが悪化する変更は採用しない
- 1 つの fixture だけを通す phrase hack は採用しない
- 機序説明ができない改善は採用しない

### 採用条件

改善は、次を同時に満たすときだけ採用する。

1. 対応テストが通る
2. 少なくとも 2 軸以上のスコア改善に寄与する
3. `thread / residual / contact / boundary / green` のどれかの機序で説明できる
4. generic fallback の混入を増やさない

### 1 サイクルの手順

1. もっとも低い軸を選ぶ
2. その軸に対応する fixture / test を追加する
3. 機序で説明できる最小修正だけを入れる
4. 近傍テストを回す
5. snapshot と target を更新する

今の優先軸は、
`数ターン継続 64 -> 76`
である。
