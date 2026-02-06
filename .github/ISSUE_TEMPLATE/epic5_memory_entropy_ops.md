---
name: "Epic 5: 記憶熱力学（Entropy/Memory Ops）"
about: "Landauer拡張・Physical AI制約を“運用規約”としてeqnet coreに導入する"
title: "[Epic 5] 記憶熱力学（Entropy/Memory Ops）"
labels: ["epic", "memory", "observability", "safety"]
assignees: []
---

## 背景 / 位置づけ
- 数式の導入自体が目的ではない。
- 導入対象は「記憶操作の不可逆性」と「知能処理のエネルギー上限」の **運用規約**。
- 本Epicは Epic 4 と矛盾せず、`nightly = 代謝` を制度化する補助線とする。

## スコープ
### 対象操作（Memory Ops）
- `add`
- `summarize`
- `forget`
- `defrag`

### 新規に追加するトレースキー（契約）
**Entropy / Irreversibility**
- `memory_entropy_delta`
- `entropy_cost_class` (`LOW|MID|HIGH`)
- `irreversible_op` (`true|false`)
- `entropy_budget_ok` (`true|false`)

**Phase-dependent valuation**
- `memory_phase` (`exploration|stabilization|recovery`)
- `phase_weight_profile` (string / config-key / profile-id)
- `value_projection_fingerprint` (string)

**Virtual Energy Budget**
- `energy_budget_used`
- `energy_budget_limit`
- `budget_throttle_applied` (`true|false`)

## 非スコープ（明確化）
- 熱力学の厳密導出 / 物理的ジュール換算はやらない
- 実装上の閾値のハードコードは禁止（config/policyに分離）
- E2Eでの“完全最適化”ではなく **silent failure を封じる**のが目的

---

# Epic 5 タスク分解

## 5-0 契約凍結（0.5日）
### 目的
trace/nightly/e2e に必要な必須キーを固定し、後続の実装を疎結合にする。

### DoD
- 下記キーが「必須」として契約化される（欠落は `YELLOW` または `WARN`）
  - `memory_entropy_delta`
  - `memory_phase`
  - `energy_budget_used`
  - `budget_throttle_applied`
- どの操作（add/summarize/forget/defrag）でも同じスキーマで出る

### 実装メモ
- `TraceEvent` / `AuditEvent` のスキーマを1箇所に集約
- 例外時でも key が欠落しないよう default は **policy** 側で供給する（平文埋め込み禁止）

---

## 5-1 nightly defrag を「代謝監査」として制度化（0.5-1日）
### 目的
nightly defrag を単なる最適化ではなく「代謝（収支）」として監査可能にする。

### 要件
- defrag 実行時に必ずエントロピー収支を記録
- `entropy_budget_ok` 欠落は `YELLOW`

### DoD
- nightly 実行ログに下記が必ず含まれる
  - `memory_entropy_delta`
  - `entropy_cost_class`
  - `irreversible_op`
  - `entropy_budget_ok`
- `irreversible_without_trace` を警告化（traceが無い不可逆操作を検出）

---

## 5-2 フェーズ依存評価（固定重み禁止）の導入（0.5-1日）
### 目的
同一記憶でもフェーズで重みを変える。評価軸の固定を禁止し「状態依存」を担保する。

### 要件
- `phase_weight_profile` を `exploration/stabilization/recovery` で切替
- フェーズ遷移時に「重み更新が未反映」なら `YELLOW`

### DoD
- フェーズ変更日の trace で以下が追える
  - `memory_phase` の変化
  - `phase_weight_profile` の更新
  - `value_projection_fingerprint` の更新（=評価軸が変わった証跡）

---

## 5-3 仮想エネルギー上限 → OutputControl 連携（1日）
### 目的
知能処理に仮想エネルギー上限を導入し、超過時は出力を慎重側へ寄せる。

### 管理対象（例）
- 新規記憶生成量
- linking件数
- 内省ループ回数

### 要件
- `energy_budget_used` / `energy_budget_limit` を計測
- 超過時 `budget_throttle_applied=true`
- `OutputControl` を慎重側へ寄せる（温度/想起/冗長性）

### DoD
- budget超過が **silent failure にならない**
- `response_gate_v1`（または同等）で制御が trace で追える
  - 「なぜ慎重側になったか」の理由がイベントに残る（ただし秘匿情報は載せない）

---

## 5-4 E2E封印（0.5日）
### 目的
シナリオ注入で `defrag実行` `budget超過` `phase切替` を再現し、nightly + trace の双方で成立を検証。

### DoD
- テストシナリオで以下を再現できる
  - defrag 実行 → 収支がログに残る
  - budget超過 → `budget_throttle_applied=true`
  - phase切替 → 重み更新が trace で追える
- nightly と trace の両方で整合

---

# 監査ルール（全体）
## YELLOW 条件
- 必須キー欠落（契約違反）
- フェーズ遷移時に `phase_weight_profile` が更新されていない
- defrag 実行時に `entropy_budget_ok` が欠落
- 不可逆操作なのに trace が追えない（`irreversible_without_trace`）
- `budget_throttle_applied=true` なのに `energy_budget_used < energy_budget_limit`

## Notes
- 値の閾値や分類ルールは `policy/config` に分離し、実装に直書きしない
- ログに秘匿（個人情報・平文キー等）を載せない
