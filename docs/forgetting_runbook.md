EQNet Nightly Forgetting Runbook
(運用チェックリスト)

目的
- 忘却(Forgetting)とDefragの運用確認を、最短の観測点で判定する
- Nightly限定・疎結合・設定駆動を崩さない

前提
- NightlyはHub経由で実行する (単体実行はforgetting/defragが出ない場合あり)
- 判定はnightly.jsonとforgetting_items.jsonlだけで完結する

1. 直近Nightlyの合否判定 (最小)
確認するファイル
- reports/nightly.json
- reports/forgetting/forgetting_items.jsonl

合格条件 (nightly.json)
- forgetting.status == "applied"
- forgetting.delta_stats が存在
- forgetting.floors_applied / forgetting.consent_overrides / forgetting.monument_locks が存在
- forgetting.source_counts が存在 (hubとheart_os_sessionが両方 > 0)

合格条件 (forgetting_items.jsonl)
- 行が存在する
- 1行あたりに以下キーがある
  - w_before / w_after / delta
  - factors (recall / affect / interference)
  - guard (consent_override / monument_lock / monument_floor)
  - sources (hub / heart_os_session)

2. 典型的な不合格パターンと対応
ケースA: delta_statsが0で張り付く
- 入力証拠が全部0 (factorsが全て0)
- 対応: moment_log_path と source_paths.heart_os_session の実ログを確認

ケースB: heart_os_sessionが混ざらない
- source_countsのheart_os_sessionが0
- 対応: affect_day_mode と affect_fields の設定を確認

ケースC: affect_samplesが0
- 日付のズレ (当日ログが無い)
- 対応: affect_day_mode を latest_available に切替

3. 監査用の貼り方テンプレ
nightly.json (5キーだけ貼る)
forgetting:
  status: ...
  delta_stats: ...
  floors_applied: ...
  consent_overrides: ...
  monument_locks: ...
  source_counts: ...

forgetting_items.jsonl (先頭1〜3行だけ、内容は伏せてOK)
{"w_before": ..., "w_after": ..., "delta": ..., "factors": {...}, "guard": {...}, "sources": ...}

4. Defrag (no-op) の確認
- defrag.enable: false -> nightly.json に {"status": "disabled"}
- defrag.enable: true -> {"status": "no_op", "reason": "placeholder"}

5. Monument接続floorテスト
- forgetting.monument_floor_test: true で実行
- nightly.json の monument_floor_test を確認
  - status: applied
  - delta_p_min / delta_p_mean / delta_p_max
  - delta_p_top (IDのみでOK)

8. Defrag Stage1（観測-only）の確認
目的
- 記録を触らず、断片化の兆候（重複/競合/参照候補/rollup候補）を夜間に観測する

出力（nightly.json）
- defrag.status == "observed"
- defrag.mode == "observe"
- defrag.observe.* が存在（値が0でも空でもOK）

安全性（必須）
- mode=observe では 記録・重み・導線（index含む）を変更しない

id_privacy（運用上の扱い）
- mask: 人が読めるが復元不能
- hash: 同一性追跡向け（推奨）
- raw: 原則禁止（デバッグ時のみ）

解釈（数値で裁かない）
- duplicate_cluster_count 増加: 似た記憶が増えている兆候
- conflict_cluster_count 増加: 干渉が増えている兆候
- reference_candidate_count 増加: Reindex（Stage2）の価値が上がっている兆候
- rollup_candidate_count 増加: Rollup（Stage4）の価値が上がっている兆候

スモークチェック（最小）
- defrag.enable: true で Nightly を1回回し、defrag.observe.* が出ること
- defrag.enable: false で Nightly を1回回し、status が "disabled" になること
- id_privacy が設定どおりに効くこと（生IDが混ざらない）

6. テストコマンド (実行用)
Nightly (Hub経由)
```powershell
$env:PYTHONPATH='.'; python ops\nightly.py
```

Nightly (Hub経由 + forgetting/defrag/monument_floor_test を明示)
```powershell
$env:PYTHONPATH='.'; @'
import yaml
from pathlib import Path
from ops import nightly
from emot_terrain_lab.hub.hub import Hub

cfg = yaml.safe_load(Path("config/runtime.yaml").read_text(encoding="utf-8")) or {}
hub = Hub(cfg)
nightly.run(hub, cfg)
'@ | python -
```

Schema テスト
```powershell
python -m pytest tests\test_nightly_schema.py
```


7. よくあるNG例 (最短チェック)
- status: applied だが delta_stats.max == 0 -> 入力証拠ゼロ (source_counts / 日付 / affect_day_mode を確認)
- source_counts が片側のみ -> path / 日付フォールバック / input_sources を確認

補足
- test_fixture のMonument/Episode/ReplayTraceは回帰テスト資産として残して良い
- 運用時は monument_floor_test を false に戻す

8. 説明レイヤ指針（人向け・運用向け）
目的
- 内部評価指標をそのまま人向けに出さない
- 共生共鳴の前提（落ち着き・関係性）を損なわない

説明は「三つの短い理由」で固定
- 落ち着きを優先している
- 記憶とのつながりが薄い
- 予測がぶれる

禁止事項
- 単一の数値指標で良否を断定しない
- 監査用の内部指標名（例: ρ、予測残差、エントロピー）をそのまま人向けに出さない

運用ルール
- 監査は数値、対話は言葉で行う
- Nightlyの結果は要約して提示する
