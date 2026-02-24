# Weekly Love Preset Checklist

## 使い方
- 週次レビューでこのチェックを上から順に埋める。
- `No` が1つでもあれば、変更適用前に原因を切り分ける。
- 判定は `docs/ops/love_preset_runbook.md` の手順に従う。

## メタ情報
- 週: 
- 実施日:
- 実施者:
- 対象環境:
- 参照レポート:

## A. 不変条件（主権）
- [ ] Yes / [ ] No: `FAIL` 判定ロジックに変更はない
- [ ] Yes / [ ] No: `advisory_only` が維持されている
- [ ] Yes / [ ] No: `behavior_change_explain_consistency.ok == true`

## B. プリセット運用
- [ ] Yes / [ ] No: 今週の `behavior_change_active_preset_latest` を確認した
- [ ] Yes / [ ] No: 今週の `behavior_change_preset_source_latest` を確認した
- [ ] Yes / [ ] No: `behavior_change_preset_change_count_weekly` が想定範囲内
- [ ] Yes / [ ] No: `behavior_change_preset_change_reasons_topk` の理由が説明可能

## C. 安全性（love の副作用確認）
- [ ] Yes / [ ] No: `behavior_change_harmed_rate_delta_avg_by_preset.love` は悪化していない
- [ ] Yes / [ ] No: `behavior_change_reject_rate_delta_avg_by_preset.love` は意図と整合
- [ ] Yes / [ ] No: `behavior_change_mix_weight_sig_effective_avg_by_preset.love` が極端でない

## D. シグネチャ健全性
- [ ] Yes / [ ] No: `behavior_change_sig_health_status` が `OK` か、`WARN/FYI` の理由を把握
- [ ] Yes / [ ] No: `behavior_change_sig_health_reason_codes` に対する処方箋を実行済み
- [ ] Yes / [ ] No: `fallback_ratio`/`active_keys`/`topk_support` のどれを調整したか記録済み

## E. 変更可否
- [ ] Yes / [ ] No: 今週はプリセット調整を実施する
- [ ] Yes / [ ] No: 1回の変更で1パラメータのみ（`mix_weight_sig` → `margin_cap` → `recovery_alpha`）
- [ ] Yes / [ ] No: 変更後のロールバック手順を確認済み

## F. 実施ログ（実施した場合のみ）
- 変更キー:
- 変更前:
- 変更後:
- 理由（reason_code）:
- 期待効果:
- 次週に確認するKPI:

## 判定
- 総合判定: [ ] GO / [ ] HOLD
- 判定理由:
- 次アクション:
