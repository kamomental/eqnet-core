# Epic 6: Policy運用（変更管理・互換性・ロールバック）

## 目的
Policy変更を運用可能な契約として固定する。  
対象は「安全な変更」「互換性」「即時ロールバック」。

## 参照
- Issue Template: `.github/ISSUE_TEMPLATE/epic6_policy_operations.md`
- 前提: Epic 5 完了（Entropy/Memory Ops封印済み）

## 設計原則
- 変更は必ず `version/profile` で識別する。
- 変更は必ず `shadow -> canary -> on` の順で通す。
- rollback導線を先に作り、後から変更を入れる。
- 平文理由ではなく `reason_codes` を使う。

## 運用契約（最初に固定する4点）
### 1) 互換性ルール
- Non-breaking:
  - 新キー追加
  - しきい値変更（内部運用規約の範囲）
  - profile追加
- Breaking:
  - 既存キー削除
  - 同キーの意味変更
  - profile名変更
  - fingerprint生成規則変更

### 2) `policy_version` の解決順序（Resolution Order）
1. runtime override（テスト専用、WARN）
2. `EQNetConfig` 指定
3. DEFAULT（安全側）

- 解決結果は必ず trace に記録:
  - `policy_version`
  - `policy_source`（`OVERRIDE|CONFIG|DEFAULT`）

### 3) ロールバック手順
- 直前バージョンへ戻す手順を固定する。
- ロールバック時の確認コマンドを固定する。
  - `pytest tests/e2e/test_e2e_hub_closed_loop.py -k entropy_memory_ops_e2e_sealed -q`

### 4) 実行前 Policy 検証（lint）
- 例:
  - `enabled_metrics` が空でない
  - thresholds が単調
  - 参照profileが存在
- 失敗時の扱いを固定:
  - 起動拒否、または `WARN/YELLOW`

## 閾値変更の定義（重要）
本Epicで扱う閾値変更は、外部物理量・環境依存パラメータを前提としない。  
内部運用規約（policy）としての分類・制御境界の変更のみを指す。

## 最小契約キー
- `policy_version`
- `policy_profile_id`
- `policy_compat_version`
- `policy_source`
- `rollout_mode`
- `rollback_ready`

## 実装順（半日〜1日）
1. `6-0` 契約凍結
2. `6-1` 互換チェック
3. `6-2` 段階展開
4. `6-3` ロールバック導線
5. `6-4` E2E封印

## 完了判定
- trace/nightly/e2e で必須キー欠落なし
- rollout mode 遷移が監査で追跡可能
- rollback を1回の操作で再現可能
- 封印E2Eがグリーン
