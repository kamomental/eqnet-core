# Evaluation Operating Policy

## 目的
`Inner OS` / `EQNet` の改善を、単一ベンチや単一 fixture への最適化ではなく、
複数軸の能力改善として進めるための運用ルールを定義する。

この文書は、以下を防ぐために置く。

- 既知の fixture だけに効く phrase hack
- wording の継ぎ足しだけで見かけの点を上げること
- 1 つの評価軸だけを上げて、live の自然さや他軸を悪化させること
- state / packet / act の機序改善ではなく、surface の偶然でスコアを稼ぐこと

---

## 基本原則

### 1. 単一指標で合格にしない
改善は、少なくとも以下の複数軸で確認する。

- deep disclosure
- thread reopening / continuity
- bright / light topic
- live-like quick exchange
- general human-facing surface

どれか 1 つだけ改善し、他が悪化する変更は採用しない。

### 2. テストと live の両方を見る
`pytest` の通過だけでは採用しない。
最低限、次の 3 面で確認する。

- 近傍回帰テスト
- holdout fixture
- live 実走

### 3. phrase hack を採用しない
次のような修正は、それ単体では採用しない。

- fixture に出た 1 語だけを潰す wording 修正
- locale 候補の追加だけで点を上げる修正
- bridge guard に禁止語を足すだけの修正

これらは、必ず packet / act / discourse shape / selector のどこに効くのか
機序で説明できるときにだけ採る。

### 4. raw と final の両方を見る
表出評価では、`final` だけでなく `raw` も観測する。

- `raw` が一般論 / 窓口文体 / ワーク化へ滑っていないか
- `final` が compaction だけで救っていないか
- `planned sequence` 自体が妥当か

### 5. holdout を残す
既知 fixture と同じ言い回しだけで改善を判定しない。
次のいずれかを holdout として毎回残す。

- 別 wording の深い話
- 別 wording の reopening
- 明るい話題の継続
- live での短い comment pickup

---

## 評価面

### A. Structure
state から final までの機序が通っているかを見る。

- `state -> packet -> act -> discourse shape -> surface`
- `surface_context_packet` に必要な状況が入っている
- `act` が会話位相と合っている
- `discourse shape` が話題の深さ / 明るさと合っている

### B. Surface
人が読んだときの自然さを見る。

- 一般論化しない
- 窓口化しない
- ワーク化しない
- 非口語の説明語に寄らない
- 1 文目で意味が通る

### C. Continuity
複数ターンでの継続感を見る。

- thread anchor が保たれる
- residual が caution / hesitation / reopening に効く
- generic continuation に戻らない
- bright topic で deep-talk 骨格に沈まない

### D. Live
短く、速く、場に追従できるかを見る。

- 2 文以内で返せる
- tempo が落ちすぎない
- comment pickup や軽い話題で慎重すぎない
- audience / group の空気を壊さない

---

## 不採用条件
次のいずれかに当てはまる変更は採用しない。

- 1 つのテストだけ上がり、holdout か live が悪化する
- `raw` は悪化しているのに `final` の compaction だけで通している
- wording の継ぎ足しで、機序の説明ができない
- bright / live / continuity のどれかが regress する
- `Inner OS` の state ではなく LLM prompt だけに依存している

---

## 採用条件
改善は次の条件を満たしたときに採る。

1. 近傍回帰テストが通る
2. holdout fixture で regress しない
3. live 実走で自然さが悪化しない
4. `packet / act / discourse shape / surface` のどこに効いたか説明できる
5. 既存の deep / reopen / bright / live のうち、少なくとも対象軸が改善する

---

## 1 サイクルの進め方

1. もっとも弱い軸を 1 つ選ぶ
2. その軸に対する holdout fixture を用意する
3. wording ではなく機序で説明できる最小修正を入れる
4. 近傍テストを回す
5. holdout を回す
6. live 実走で raw / planned / final を比較する
7. `evaluation_snapshot` と `evaluation_targets` を更新する

---

## 現時点の優先
2026-03-28 時点では、次を優先する。

1. multi-turn continuity
2. bright / light topic の自然さ
3. expression の `act / discourse shape / surface realization` 分離

deep disclosure の改善は維持しつつ、bright / live を同じ表出系の連続軸で扱えるようにする。
