---
name: "Epic 6: Policy運用（変更管理・互換性・ロールバック）"
about: "Policy変更を安全に運用するための契約と監査を導入する"
title: "[Epic 6] Policy運用（変更管理・互換性・ロールバック）"
labels: ["epic", "policy", "operations", "observability"]
assignees: []
---

## 背景
- Epic 5で `契約 -> 監査 -> E2E封印` の土台は完成した。
- 次は policy 変更時の事故（意味変化、互換崩れ、ロールバック不能）を運用で防ぐ。

## 目的
- Policy変更を「追跡可能・比較可能・即時ロールバック可能」にする。

## スコープ
- policy versioning
- compatibility check
- rollout/rollback control
- nightly drift audit

## 非スコープ
- モデル性能最適化そのもの
- 新しい推論アルゴリズム導入

---

## 6-0 契約凍結
### 必須キー
- `policy_version`
- `policy_profile_id`
- `policy_compat_version`
- `rollout_mode` (`shadow|canary|on`)
- `rollback_ready` (`true|false`)

### DoD
- trace/nightly/e2e で欠落時 `YELLOW`

## 6-1 互換チェック
### 要件
- 旧policyとの差分を機械判定
- 互換破壊時は `YELLOW` 以上

### DoD
- `compat_check_result` が監査に残る

## 6-2 段階展開（shadow/canary/on）
### 要件
- rollout mode を policyで切替
- mode を trace で必ず観測

### DoD
- mode切替履歴が day単位で追える

## 6-3 ロールバック導線
### 要件
- rollback対象の version/profile を即時指定可能
- rollback成功/失敗を監査へ出力

### DoD
- `rollback_ready=true` が保証される

## 6-4 E2E封印
### シナリオ
- `shadow -> canary -> on -> rollback`

### DoD
- mode遷移と rollback が trace/nightly で整合

---

## YELLOW 条件
- 必須キー欠落
- 互換チェック未実施
- rollout mode と実際適用の不一致
- rollback不能状態
