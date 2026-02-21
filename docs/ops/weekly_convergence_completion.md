# Weekly Convergence Completion

## Purpose
- このドキュメントは、今回スコープの「実装収束」を完了判定するための最小運用手順を固定する。
- 判定は「実装があるか」ではなく「12週の観測に基づいて意思決定ループが閉じたか」で行う。

## Completion Criteria
1. `weekly_calibration_YYYY-WW.json` が12週分そろっている。
2. 12週サマリを生成して結果を確認している。
3. 結果に対して「見直す条件 / 維持する条件（none）」を1セット確定している。

## Commands
```bash
python scripts/summarize_weekly_calibration.py --weekly-dir reports --weeks 12
```

## Reading Order
1. `none_streak.current / max`
2. `delta_spike_weeks`
3. `proposal_proxy.follow_rate_proxy / match_rate_proxy`

## Decision Rules (Initial)
- 維持（none継続）:
  - `none_streak.current >= 2`
  - かつ `delta_spike_weeks` が直近2週で発生していない
- 見直し候補:
  - `delta_spike_weeks` が直近4週で2回以上
  - または `proposal_proxy.match_rate_proxy < 0.5`

## Weekly Record Template
- 判定: `maintain` / `recalibrate`
- 理由: 1-2行
- 変更対象（recalibrate時）:
  - 境界幅
  - 注意配分
  - 時間感度
- 備考:

## Notes
- `proposal_proxy` は近似指標。厳密な採用・一致判定ではない。
- `schema_version` 不一致ファイルは集計から自動スキップされる。
