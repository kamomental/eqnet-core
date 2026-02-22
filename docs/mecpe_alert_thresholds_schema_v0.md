# MECPE Alert Thresholds schema v0 (frozen)

## 目的
- 週次の `mecpe_alert`（流量異常）の閾値を、コード埋め込みではなく契約ファイルとして固定する。
- 判定時に使った閾値を `thresholds_snapshot` に残し、後から再現・説明できる状態を維持する。

## 対象ファイル
- `configs/mecpe_alert_thresholds_v0.yaml`
- 環境変数 `EQNET_MECPE_ALERT_THRESHOLDS` で読み込み先を上書き可能。

## スキーマ（v0）
- `schema_version`: `mecpe_alert_thresholds.v0`
- `approved_count_min`: int
- `pending_ratio_warn`: float
- `pending_ratio_alert`: float
- `oldest_pending_age_warn_days`: int
- `oldest_pending_age_alert_days`: int

## 判定の意味
- `approved_count_min`: 分母が小さい週の誤検知抑制。
- `pending_ratio_warn/alert`: 未完了率に基づく WARN/ALERT 境界。
- `oldest_pending_age_warn_days/alert_days`: 慢性滞留（時間軸）に基づく WARN/ALERT 境界。

## Fail-safe（凍結）
- 設定ファイル未存在、または不正な内容の場合は安全デフォルトへフォールバック。
- フォールバック時も、実際に使用した値は `mecpe_alert.thresholds_snapshot` に出力される。

## デフォルト値（v0）
- `approved_count_min: 5`
- `pending_ratio_warn: 0.2`
- `pending_ratio_alert: 0.5`
- `oldest_pending_age_warn_days: 7`
- `oldest_pending_age_alert_days: 14`

## チューニング手順（推奨）
1. 過去N週の分布（approved_count, pending_ratio, oldest_pending_age_ms, latency p95）を確認。
2. WARN/ALERT を分位点（例: 80/95）または運用コスト（誤検知/見逃し）で調整。
3. `configs/mecpe_alert_thresholds_v0.yaml` を更新し、変更理由をPRに明記。
4. 反映後は `thresholds_snapshot` と発火頻度を週次で観測し、過剰発火や見逃しを再評価。

## 非機能要件
- 閾値は運用設計上の初期値であり、自然法則ではない。
- 変更は必ずレビュー対象とし、監査ログ（週次JSON）で追跡可能にする。
