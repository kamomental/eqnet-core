# Deep Disclosure Reflection

`deep disclosure` は、ユーザーが未表出の痛みや怖さを短く開示したときに、
generic な共感文や保護文より先に、内容反映を 1 文置くための表出ブリッジです。

## 目的

- `そう感じていること、わかります` のような汎用共感に落ちすぎるのを防ぐ
- `助けてほしかった / まだ言えていない / どう見られるか怖い / 自分を責めてしまう`
  のような核を、最初の一文で受け取る
- その後の問いは 1 つだけ、低圧で置く

## 現在の扱い

- 実装位置:
  - `inner_os/expression/content_policy.py`
  - `emot_terrain_lab/hub/runtime.py`
  - `locales/ja.json`
- shallow な `generic empathy + clarification` より前に評価される
- short compaction では
  - `reflection`
  - `gentle question`
  の 2 文を優先する

## 例

- `本当は、あのとき助けてほしかったって、まだ言えていないんです。`
  - `助けてほしかったのに、それをまだ言えないままなんですね。`
  - `いちばん引っかかっているのは、そのひと言を飲み込んだところですか。`

- `話すと、相手の見方が変わりそうで怖いんです。`
  - `言ったあとにどう見られるか、その怖さがまだ強いんですね。`
  - `いちばん大きいのは、言ったあとにどう見られるかの怖さですか。`
