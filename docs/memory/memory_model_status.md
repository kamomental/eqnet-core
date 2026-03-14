# 記憶モデル現状整理

## 目的

この文書は、`emotional_dft` における記憶まわりの実装を、

- 何がすでにあるか
- 何が概念として入っているか
- 何が未整理か
- 今後どこを揃えるべきか

の観点で整理するためのものです。

ここでの主眼は新規理論の追加ではなく、既存コードの意味を揃えることにあります。

## 基本認識

現状の `emotional_dft` には、記憶の実装はすでに相当量あります。
したがって、今やるべきことは「記憶機能の新設」よりも、

- 既存の記憶層の責務を明確にする
- 想起、再点火、再構成、検証の区別を明示する
- `inner replay` と記憶の境界を揃える

ことです。

## 現在ある主要実装

### 1. 多層記憶

ファイル:
- `emot_terrain_lab/terrain/system.py`
- `emot_terrain_lab/terrain/memory.py`

現在の中心は `EmotionalMemorySystem` です。
この中には以下の層がすでにあります。

- `l1`: raw experience
- `l2`: episodic memory
- `l3`: semantic memory
- `field`: emotional field
- `story_graph`
- `memory_palace`

### 2. Memory Palace

ファイル:
- `emot_terrain_lab/terrain/memory_palace.py`

`MemoryPalace` は、

- ノード位置
- trace
- label
- qualia state

を持ち、場所や近接性や情動負荷に近いものを扱っています。

### 3. 想起参照系

ファイル:
- `emot_terrain_lab/memory/reference_helper.py`
- `emot_terrain_lab/memory/recall_policy.py`

ここにはすでに以下があります。

- 参照解決 `resolve_reference`
- 候補検索 `search_memory`
- リプレイ評価 `run_replay`
- 想起応答 `compose_recall_response`
- 想起入口 `handle_memory_reference`
- cue ベースの想起表現 `render_recall_cue`

### 4. 再点火トレース

ファイル:
- `emot_terrain_lab/hub/recall_engine.py`

`RecallEngine` は、

- anchor cue の検出
- activation chain の構築
- confidence curve の記録
- replay event の保存

を行います。

### 5. テスト

ファイル:
- `tests/test_memory_reference_helper.py`
- `tests/test_recall_engine.py`

ここには、

- `2019 Kyoto trip`
- `walk`
- `rest`
- `anchor:bakery`

のような想起・再点火に関わるケースがすでに含まれています。

## 現時点での評価

### すでにあるもの

- 多層記憶の発想
- 場所アンカー的な保存
- 想起参照
- cue ベース応答
- 再点火トレース
- 想起と監査の一部

### まだ揃っていないもの

- 各層の責務が文書として統一されていない
- `reference_helper` と `recall_engine` と `memory_palace` の関係が明文化されていない
- `observed` / `reconstructed` / `verified` の保存上の区別が弱い
- 地理的手がかりから経路・人物・感情が連鎖する厚い想起は、概念的には近いが実装上はまだ分散している
- 一部ファイルに文字化けがあり、意図が読み取りにくい

## このプロジェクトでの記憶の定義

このプロジェクトでは、記憶を単なる検索対象ではなく、

- 現在の手がかりで再点火されるもの
- 感情や人物関係と一緒に立ち上がるもの
- 場所差分や後年の検証で更新されうるもの

として扱うのが妥当です。

## 記憶層の実務的整理

今後は最低でも以下の区別を文書と実装で揃えるべきです。

### observed

実際に観測・発話・入力されたもの。

### episodic

出来事としてまとまった記憶。

### associative

想起を起こす手がかり。

### reconstructed

後から再構成された意味や想起内容。

### verified

後年の確認や外部検証で補強された内容。

## inner replay との関係

`inner replay` は重要ですが、記憶そのものではありません。

役割の整理:

- 記憶は過去側の再点火と再構成
- `inner replay` は未来側の仮想と予測

したがって、

- 記憶: `Now -> Past`
- `inner replay`: `Now -> Future`

という違いがあります。

## いまの最善方針

いま必要なのは、記憶機能を新しく作り直すことではなく、
既存のものを次の方針で揃えることです。

1. `EmotionalMemorySystem` を記憶の正規所有者として扱う
2. `reference_helper` は想起入口として扱う
3. `recall_engine` は再点火観測器として扱う
4. `memory_palace` は場所アンカー兼 qualia 付き記憶座標として扱う
5. `observed` / `reconstructed` / `verified` を保存上も区別する

## 当面の実装タスク

### 優先

1. 記憶系ファイルの責務を文書上で固定する
2. `reference_helper.py` の文字化けを安全に修正する
3. `observed` / `reconstructed` / `verified` の区別をログまたは保存スキーマに追加する

### 次点

1. 場所手がかりからの連鎖想起を `memory_palace` と `recall_engine` の間で明示化する
2. 想起応答時に source class をより厳格に区別する
3. 後年検証を `verified overlay` として重ねる保存モデルを作る

## 対応表

| 機能 | 既存ファイル | 現在の役割 |
|---|---|---|
| 多層記憶の統合 | `emot_terrain_lab/terrain/system.py` | 記憶全体の所有と保存 |
| 感情記憶 / L1-L3 | `emot_terrain_lab/terrain/memory.py` | 記憶層の基礎 |
| 場所アンカー | `emot_terrain_lab/terrain/memory_palace.py` | 場所的・qualia 的アンカー |
| 想起入口 | `emot_terrain_lab/memory/reference_helper.py` | 記憶参照、候補検索、応答生成 |
| 想起表現 | `emot_terrain_lab/memory/recall_policy.py` | cue ベースの表出制御 |
| 再点火観測 | `emot_terrain_lab/hub/recall_engine.py` | anchor cue と activation trace |

## まとめ

`emotional_dft` の記憶モデルは、まだ未完成ではあるものの、
「再点火」「多層記憶」「場所アンカー」「想起の表現制御」という重要な要素はすでに入っています。

したがって今後の方針は、

- 新設より再整理
- 思想追加より責務固定
- 想起の美しさより事実区分の明確化

を優先するのが妥当です。