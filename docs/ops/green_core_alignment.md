# Green層とEQNet Core層の整合方針（ADR相当）

## 目的
Green関数ベースの情動地形（`emot_terrain_lab`）と、監査可能な制御コア（`eqnet core`）を矛盾なく共存させる。

本書は「どちらを捨てるか」ではなく、「役割を分けて接続する」ための運用原則を固定する。

## 現状整理
- Green層は存在する:
  - `emot_terrain_lab/mind/green.py`
  - `emot_terrain_lab/core/green_kernel.py`
  - `tools/tune_green_kernel.py`, `tools/analyze_green_impulse.py`
- Core層は存在する:
  - `scripts/run_nightly_audit.py`
  - `eqnet/runtime/future_contracts.py`
  - `eqnet/runtime/sync_realtime.py`
  - replay/diff/gate 系
- 問題は機能欠落ではなく、層間の配線不足である。

## 設計原則（不変条件）
1. 役割分離:
   - Green層は「応答地形の推定・観測」を担う。
   - Core層は「承認・抑制・監査可能な判定」を担う。
2. 判定主権:
   - 採択/却下/ブロック/ダウンシフトはCore層の契約で決定する。
   - Green層は判定を上書きしない。
3. 接続段階:
   - Phase 0: record-only（監査用メトリクスとして記録）
   - Phase 1: priority補正（並び順にのみ反映）
   - Phase 2: 判定寄与を検討（別ADRと検証が必要）
4. 監査一貫性:
   - `policy_meta` / `fingerprint` / `reason_codes` / `eval_ts_ms` を必須化する。
   - replay/diff/gate で `green_on/off` 比較を可能にする。
5. 安全哲学維持:
   - 自動適用は禁止（承認ゲート維持）。
   - UNKNOWN優先・downshift優先を維持する。

## 最小接続仕様（実装前提）
- Coreへ流すGreen派生値は正規化した少数キーに限定する:
  - `green_response_score`
  - `green_decay_tau`
  - `green_mode`
  - `green_quality`
- 追記先:
  - nightly payload（record-only）
  - replay aggregate（比較可能化）
- 判定反映:
  - まず `priority_score` 補正に限定（採択判定には使わない）。

## Greenが出せるもの / 出せないもの
| 区分 | 可否 | 内容 |
|---|---|---|
| 推定スコア | ✅ | `green_response_score` などの観測・推定値 |
| 候補提案 | ✅ | 承認前提の提案候補（record-only/priority補助） |
| 説明文 | ✅ | 参考説明（reason_codesに紐づく非クリティカル出力） |
| 品質補助 | ✅ | `priority_score` の補助係数（段階接続の範囲内） |
| 最終判定上書き | ❌ | approve/block/downshift/gate 判定の上書き |
| 承認の強制 | ❌ | 人間最終決定の迂回、強制採択 |
| policy直接更新 | ❌ | fingerprint更新を伴う直接反映 |
| imageryの事実化 | ❌ | 未検証仮説を fact として扱う混入 |

## 非目標
- Green層を即時に判定中核へ組み込むこと。
- 既存の承認/ゲート/監査契約を破壊する変更。

## 受け入れ条件（DoD）
1. Green接続時でも既存回帰（nightly/replay/diff/gate/sync）が維持される。
2. `green_on/off` で差分比較可能である。
3. gate基準（回帰優先）を緩めない。
4. 理由は `reason_codes` で説明可能である。

## 次アクション
1. `green_bridge` を追加して record-only 接続。
2. `config_sets A/B` に green bridge 設定を追加。
3. replay/diff に green 指標を追加し、gate通過条件を確認。

## PR最小チェックリスト
1. Green出力は `record-only` か `priority補正` までに収まっているか。
2. Core判定（gate/approve/block/downshift）を変更していないか。
3. `fingerprint` / `reason_codes` / `eval_ts_ms` が追跡可能か。
4. replayで `green_on/off` の比較結果を確認したか。
