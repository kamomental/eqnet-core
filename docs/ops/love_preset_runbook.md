# Love Preset Runbook (EQNet Core)

## 目的
- `love` プリセットを安全に運用する。
- 可塑性は `WARN` 領域でのみ扱い、`FAIL` 境界は不変とする。
- 週次レビューで「緩和が有効か/危険か」を機械可読KPIで判断する。

## 前提（不変条件）
- `diff_gate` の `FAIL` 判定は固定。プリセットで変更しない。
- 自動適用は禁止。推奨は `advisory_only`。
- 説明整合は常に監査する（`behavior_change_explain_consistency.ok`）。

## 対象パラメータ
プリセット切替で変更してよいのは以下のみ。
- `tolerance.margin_cap`
- `tolerance.recovery_alpha`
- `tolerance.mix_weight_sig`

変更してはいけない項目（例）。
- `diff_gate.thresholds.*.fail`
- `tolerance.epsilon_to_fail`
- `diff_gate` の `FAIL` 判定ロジック

## 監査で必ず見る項目
日次:
- `behavior_change_active_preset`
- `behavior_change_preset_source`
- `behavior_change_explain_consistency`

週次:
- `behavior_change_preset_change_count_weekly`
- `behavior_change_preset_change_reasons_topk`
- `behavior_change_harmed_rate_delta_avg_by_preset`
- `behavior_change_reject_rate_delta_avg_by_preset`
- `behavior_change_mix_weight_sig_effective_avg_by_preset`
- `behavior_change_sig_health_status`
- `behavior_change_sig_health_reason_codes`

## 週次レビュー手順
1. まず `love` の安全性を確認する。
   - `behavior_change_harmed_rate_delta_avg_by_preset.love` が悪化していないか。
2. 次に操作性を確認する。
   - `behavior_change_reject_rate_delta_avg_by_preset.love` の変化が狙い通りか。
3. そのうえで `sig` 有効性を確認する。
   - `behavior_change_mix_weight_sig_effective_avg_by_preset.love` が低すぎないか。
4. `sig_health` が `WARN`/`FYI` の場合は reason_code を優先して処置する。

## 調整順序（必須）
変更は 1 回に 1 変数のみ。順序は固定。
1. `tolerance.mix_weight_sig`（状況依存の効き具合）
2. `tolerance.margin_cap`（WARN 緩和幅）
3. `tolerance.recovery_alpha`（回復速度）

## 処方箋（reason_code別）
`BC_SIG_HEALTH_FALLBACK_RATIO_HIGH`
- 優先: `signature.field_sanitizers` を強化
- 次点: `tolerance.min_support_per_sig` を調整

`BC_SIG_HEALTH_ACTIVE_KEYS_NEAR_CAP`
- 優先: `signature.field_sanitizers` を強化
- 次点: `signature.fields` の次元削減
- 最後: `signature.max_keys` の見直し

`BC_SIG_HEALTH_TOP_SUPPORT_LOW`
- 優先: `window.baseline_days` の見直し
- 次点: `signature.fields` の簡素化

## 運用判定ルール（最小）
- `love` で `harmed` が悪化した週:
  - `margin_cap` を下げる。
  - 必要なら `mix_weight_sig` を下げて global 寄りに戻す。
- `love` で `reject` が下がりすぎ、かつ `harmed` も悪化:
  - `margin_cap` を先に下げる。
- `mix_weight_sig_effective` が低い週:
  - 先に `sig_health` を修復し、プリセットは触らない。

## 変更手順
1. `configs/behavior_change_v0.yaml` を更新。
2. `configs/config_sets/A/behavior_change_v0.yaml` と `configs/config_sets/B/behavior_change_v0.yaml` に同じ更新を反映。
3. `replay` で `A/B` 差分を生成し、`diff_gate` を通す。
4. 週次KPIで効果確認（最低1週）。

## ロールバック手順
- `active_preset: default` に戻す。
- `preset_source: manual` を維持。
- 変更前の config fingerprint に戻し、再度 `replay -> diff -> gate` を実行。

## 変更記録テンプレート
- 変更日:
- 変更者:
- 変更対象キー:
- 変更理由(reason_code):
- 期待効果(expected_effect):
- 1週後の結果:
- 次アクション:
