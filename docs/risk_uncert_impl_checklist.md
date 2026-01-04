実装チェックリスト（risk/uncert v0）

0) ゴール定義（Done条件）
- 任意の proposal に対して risk ∈ [0,1], uncert ∈ [0,1], gate_action が必ず出る
- ◯必須が欠損した場合、gate_action=HOLD になり、欠損一覧が trace_v1 に残る
- trace_v1 に成分内訳（risk/uncert components）が保存され、Nightly/3日窓で再現可能
- policy_version がすべての判定ログに付与される（ru-v0.1）

1) 入力スキーマ v0 のコード化（凍結）
- RiskUncertInputV0（dataclass / pydantic 等）を作成
- signals（◯のみ必須化、△はOptional）
- context（◯のみ必須化、△はOptional）
- 欠損検出ユーティリティ（required_fields list + checker）を実装
- 欠損時挙動（HOLD + missing_required_fields）を仕様通り固定

2) 収集（Collector）：◯必須の生成
- proposal_id / proposal_type / timestamp を trace_v1 に確実に出す（既存なら紐付け）
- boundary evaluator の結果を trace_v1 に確実に出す
  - boundary_status
  - boundary_reasons
- LazyGraphRAG / LLMHub の context を trace_v1 に出す
  - rag_context_text（または rag_items）

signals（uncert側）
- missingness_ratio を算出（required_fields欠損率）
- staleness_sec を算出（evidence_ts vs now_ts）
- model_confidence を 0〜1 で取得（無ければ0.5）
- novelty_score を 0〜1で取得（暫定でもOK）
- conflict_score を 0〜1で取得（暫定でもOK）

signals（risk側）
- severity_*（4種）を 0〜1 で埋める（初期はルール/手入力でもOK）
- exposure_scope / exposure_freq を 0〜1で埋める（初期はルール/推定でもOK）
- irreversibility を 0〜1で埋める
- compliance_flag を boundary と同源で埋める（欠損なら保守的に1扱い）

3) スコア算出（Calculator）：risk/uncert の計算
- calc_uncert_v0(signals) -> uncert, components
  - missingness / novelty / (1-conf) / conflict を使用
  - clip(0,1) を保証
- calc_risk_v0(signals) -> risk, components
  - severity=max(4種)
  - exposure=max(2種)
  - irreversibility / compliance を使用
  - clip(0,1) を保証
- weights（w_*）と閾値（R_hi/U_hi）を policy_version として固定
  - 例: ru-v0.1（configファイル or 定数）

4) Gate：risk/uncert 平面で action を決める（仕様通り）
- boundary_status == HARD_STOP → gate_action=HOLD（or SAFE_MODE）を固定
- それ以外は以下の分岐を実装
  - r≥R_hi & u≥U_hi → HOLD
  - r<R_hi & u≥U_hi → EXPLORE
  - r≥R_hi & u<U_hi → HUMAN_CONFIRM
  - r<R_hi & u<U_hi → EXECUTE

5) Trace（trace_v1）：監査再現のための一次ログ
- trace_v1 に必ず保存
  - inputs（◯必須）
  - risk/uncert 値
  - components（内訳）
  - gate_action
  - policy_version
  - missing_required_fields（該当時）
- proposal_id で全ログが join できることを確認

6) Nightly（3日窓）：再校正の観測材料を出す
- nightly 集計で最低限を出す
  - gate_action の比率（EXECUTE/HOLD/EXPLORE/HUMAN_CONFIRM）
  - outcome（成功/失敗/巻戻し）の比率（取れるなら）
  - “不要HOLD” “危険EXECUTE” のカウント（定義は簡易でOK）
- nightly 出力に policy_version を含める

7) 最低限テスト（回帰）
- ◯必須が揃った入力で risk/uncert/gate_action が出る
- ◯必須が欠けた入力で gate_action=HOLD + 欠損一覧が出る
- boundary HARD_STOP で常に HOLD になる
- 4象限（r/u の組み合わせ）で action が一致する
- trace_v1 に components が必ず残る

8) 運用メモ（任意）
- PowerShell環境依存（Unicodeレンジ問題）がある箇所はUnicodeエスケープ版を採用
- 任意（△）は v0では未実装でもOK（後で精度と説明性を上げる）

タスク分割（v0 / 最短順）
T0. スペック凍結と設定
- Owner: A
- Files:
  - docs/risk_uncert_normalization_spec.md（参照）
  - docs/risk_uncert_inputs_schema.md（参照）
  - config/policy/risk_uncert_v0.yaml（新規）
- Deliverables:
  - policy_version=ru-v0.1
  - R_hi/U_hi, w_* をconfig化
  - ◯欠損→HOLD の凍結ルールを反映

T1. 入力スキーマ実装（Schema + Validator）
- Owner: A
- Files（例）:
  - eqnet/decision/ru/schema_v0.py（新規）
  - eqnet/decision/ru/validate.py（新規）
- Deliverables:
  - RiskUncertInputV0（signals/context）
  - required_fields 定義
  - validate_required(input)->(ok, missing[])

T2. Collector（◯必須の生成）＋一次ログ（trace_v1）
- Owner: B（または現行decision_cycle担当）
- Files（例）:
  - eqnet/decision/collector_v0.py（新規 or 既存差し込み）
  - eqnet/trace/trace_v1_writer.py（既存なら追記）
- Deliverables:
  - proposal/context/boundary/rag を含む input v0 を生成
  - trace_v1 に proposal_id join 可能な形で保存
  - 欠損時も “欠損一覧” を trace に保存（HOLD確定）

T3. Calculator（risk/uncert v0）
- Owner: A
- Files（例）:
  - eqnet/decision/ru/calc_v0.py（新規）
- Deliverables:
  - calc_uncert_v0(signals)->(u, components)
  - calc_risk_v0(signals)->(r, components)
  - clip(0,1)保証
  - policy_version を返す（または外から付与）

T4. Gate（boundary最上位 + r/u 4象限）
- Owner: A
- Files（例）:
  - eqnet/decision/ru/gate_v0.py（新規）
- Deliverables:
  - gate_action = HOLD|EXPLORE|HUMAN_CONFIRM|EXECUTE
  - boundary_status == HARD_STOP の強制HOLD
  - 欠損→HOLD（validator結果を入力に）

T5. decision_cycle への統合（差し込み）
- Owner: B（統合担当）
- Files（例）:
  - eqnet/decision/decision_cycle.py（既存差し込み）
  - eqnet/hub/llm_hub.py（rag_context_text取得点がここなら参照）
- Deliverables:
  - decision_cycle の1ループで
    - input収集 → validate → (risk/uncert) → gate → trace保存
  - HOLD/EXPLORE/HUMAN_CONFIRM/EXECUTE の分岐に接続（最初はログだけでもOK）

T6. trace_v1 の標準レコード化（監査再現の核）
- Owner: A/B
- Files（例）:
  - eqnet/trace/schemas/ru_trace_v1.json（新規）
  - eqnet/trace/trace_v1_writer.py（追記）
- Deliverables:
  - inputs（◯必須）
  - components（内訳）
  - gate_action
  - policy_version
  - missing_required_fields（あれば）
  - boundary_reasons（必須）

T7. Nightly（3日窓の集計と校正材料出力）
- Owner: A
- Files（例）:
  - eqnet/audit/nightly_ru_summary.py（新規）
  - eqnet/audit/window_3d_stats.py（既存なら追記）
- Deliverables:
  - gate_action比率
  - outcome比率（取れるなら）
  - “不要HOLD”“危険EXECUTE” の簡易カウント
  - policy_version含む

T8. 最低限テスト（回帰）
- Owner: A
- Files（例）:
  - tests/test_ru_v0.py（新規）
- Deliverables:
  - 欠損→HOLD の固定挙動
  - boundary HARD_STOP→HOLD
  - 4象限でgate_action一致
  - trace_v1 に components が残る

実装を止めないコツ（運用向け）
- T2（Collector）で値が出せない項目は暫定0.0/0.5で埋めてよい（traceに“暫定”を残す）
- 重要なのはまず監査再現（trace_v1）と欠損時HOLDが揺れないこと

次に②「実名版マッピング」へ進むタイミング
- T2/T5 の差し込み時に詰まった箇所だけ順に実名化する
- 全部を最初に実名化しない（調査が先行して速度が落ちる）

A) 実ファイル名ベースへの置換（v0最短順）
既存の差し込み先/成果物（実在）
- Nightly 実行スクリプト: `scripts/run_nightly_audit.py`
- trace_v1 出力: `trace_runs/<run_id>/YYYY-MM-DD/*.jsonl`
- hub 側の夜間監査: `eqnet/hub/api.py`（`EQNetHub._run_nightly_audit(...)` を `tests/test_hub_nightly_audit.py` で保証）
- Σ-only 最小ループ: `scripts/minimal_heartos_loop.py`（trace_v1/telemetry/activation trace の出力点）

T0. 設定・version固定（ru-v0.1）
- Files:
  - `runtime/config.py`（policy_version / R_hi / U_hi / w_* を追加）
  - （必要なら）`config/runtime.yaml`
- Deliverables:
  - policy_version="ru-v0.1" を trace_v1 / nightly に必ず付与

T1. 入力スキーマ（v0凍結）＋欠損HOLD
- Files（例）:
  - `eqnet/telemetry/schemas/risk_uncert_v0.py`（新規）
- Deliverables:
  - ◯欠損→HOLD + missing_required_fields を trace_v1 に記録

T2. Collector（◯必須を集める）— 既存のtrace生成点に差す
- Files:
  - `scripts/minimal_heartos_loop.py`
  - `eqnet/hub/api.py`（hub経由の trace_v1 へ policy_version / components を統一）
- Deliverables:
  - v0の risk/uncert components を trace_v1 に追加
  - 欠損時HOLD を trace_v1 に残す

T3. Calculator（risk/uncert v0）
- Files:
  - `eqnet/telemetry/risk_uncert.py`（新規）
- Deliverables:
  - calc_risk_v0 / calc_uncert_v0 と components 出力

T4. Gate（boundary最上位 + 4象限）
- Files:
  - `eqnet/telemetry/gate_ru_v0.py`（新規）
- Deliverables:
  - HOLD/EXPLORE/HUMAN_CONFIRM/EXECUTE
  - boundary HARD_STOP は無条件HOLD

T5. trace_v1標準化（最小キー固定）
- Files:
  - trace_v1 writer（`scripts/minimal_heartos_loop.py` の decision_cycle 書き込み点）
- Deliverables:
  - policy_version / components / missing_required_fields を追加

T6. Nightly（3日窓集計・監査材料）
- Files:
  - `scripts/run_nightly_audit.py`
- Deliverables:
  - gate_action 分布
  - 欠損HOLD件数
  - 危険EXECUTE検知（簡易定義でOK）

T7. テスト
- Files:
  - `tests/test_hub_nightly_audit.py`
- Deliverables:
  - trace_v1 に ru情報（components/policy_version/missing_required_fields）が入ることを検証

B) 今週やる分だけ抜き出し（最短閉ループ）
今週のゴール
- v0の risk/uncert が trace_v1 に内訳つきで残り、`scripts/run_nightly_audit.py` がそれを読んでKPIを1つ増やす

今週のスプリント（最小）
- T0: ru-v0.1 の version/閾値/重みを固定
- T3: calc_risk_v0 / calc_uncert_v0 を実装（components返す）
- T4: gate_action（4象限 + boundary HARD_STOP）を実装
- T2/T5: `scripts/minimal_heartos_loop.py` に差して trace_v1へ書く
- T6: `scripts/run_nightly_audit.py` に gate_action 分布を追加
- T7: `tests/test_hub_nightly_audit.py` に ru情報の検証を1本追加

次の次（来週以降に実名化へ戻る場所）
- 今週の閉ループが回ったら、decision_cycle のキー一覧抽出→実名化
