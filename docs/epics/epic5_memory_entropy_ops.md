# Epic 5: 記憶熱力学（Entropy/Memory Ops）

## 目的
Landauer拡張・Physical AI制約を、理論実装ではなく運用契約として `eqnet core` に制度化する。

## 参照
- Issue Template: `.github/ISSUE_TEMPLATE/epic5_memory_entropy_ops.md`
- 運用母体: `docs/eqnet_life_gap_operating_model.md`

## 設計上の立場（事実ベース）
Epic 5 は理論を実装しない。  
理論が将来差し替わっても壊れないように、運用の不可逆性を固定する。

## 非っぽさ判定（レビュー基準）
### 問題になる条件
- 数値の意味が将来も同一である保証がない
- 後から理論を差し替えられない構造になっている

### 失敗パターン
- ケースA: 分類の意味が時系列で変質する
  - `entropy_cost_class=HIGH` の意味が時期で変わる
  - 変更事実がログで検出できない
- ケースB: 上限が運用者依存で揺れる
  - `energy_budget_limit` の変更理由が追跡できない
  - 計測と出力制御が分離していて因果が見えない

### Epic 5 が回避している構造
- 契約凍結（5-0）で必須キーを固定
- 欠落を `YELLOW` で検出
- `phase_weight_profile` / `value_projection_fingerprint` で時間的同一性を追跡
- 仮想エネルギー（unitless） + ハードコード禁止
- `OutputControl` 連携を理由コードで追跡可能

## 禁止する説明文（レビューNG）
- 「理論上限だから」
- 「熱力学的に正しいから」
- 「Landauer的に避けられないから」

## 許可する説明文（レビューOK）
- 契約
- 監査
- 追跡可能性
- silent failure 封印

## 決定ログ
### 2026-02-06
- Epic 5 を運用対象へ採用
- `nightly = 代謝` を監査基準として採用
- 監査整合ルールを追加
  - `budget_throttle_applied=true` かつ `energy_budget_used < energy_budget_limit` は `YELLOW`
