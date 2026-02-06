# Issue Template運用メモ

## 原則
- `.github/ISSUE_TEMPLATE/*.md` は起票フォーム専用とする。
- 実装メモ、レビューコメント、運用ログは `docs/` 側へ分離する。

## 配置ルール
- Epic本体設計: `docs/epics/`
- Runbook: `docs/runbooks/`
- テンプレ本文にチャットログや進捗報告を混在させない。

## Epic 5 適用
- 起票テンプレ: `.github/ISSUE_TEMPLATE/epic5_memory_entropy_ops.md`
- 設計ログ: `docs/epics/epic5_memory_entropy_ops.md`
- Policy変更後チェック（必須）:
  - `pytest tests/e2e/test_e2e_hub_closed_loop.py -k entropy_memory_ops_e2e_sealed -q`

## Epic 6 適用
- 起票テンプレ: `.github/ISSUE_TEMPLATE/epic6_policy_operations.md`
- 設計ログ: `docs/epics/epic6_policy_operations.md`

## Epic 5 最終チェック（運用）
- 契約キー一覧（必須＋補助）を `trace/nightly` で確認する。
- YELLOW条件一覧（`nightly_audit` 実装済み）を確認する。
- sealed E2E の入口を固定する。
  - `pytest tests/e2e/test_e2e_hub_closed_loop.py -k entropy_memory_ops_e2e_sealed -q`

## ドキュメント専用PRの固定文言
- `code changes: none`
- `tests: not run (doc-only)`

## 更新手順
1. 先に `docs/epics/*` へ仕様変更理由を記録する。
2. 次に issue テンプレへ反映する。
3. 必須キー変更時は trace/nightly/e2e の契約テストを更新する。
