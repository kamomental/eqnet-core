signals/context 洗い出し（v0: 最小運用スキーマ）

原則
- signals: 観測/計測/状態（できれば数値、無ければラベル→数値化）
- context: 根拠（RAG）・履歴・制約・メタ（判断の説明に使う）
- risk/uncert の成分は、このスキーマから必ず計算可能にする（監査で再現）

A. signals（decision入力の一次観測）
A1. 観測品質（uncertに直結）
- missingness_ratio: 必要フィールド欠損率（0〜1）
- staleness_sec: 観測の鮮度（秒）→ 正規化で 0〜1
- source_count: 独立ソース数（センサ/ログ/人の報告）
- conflict_score: 根拠間矛盾（0〜1）
- model_confidence: 推定/分類の確信度（0〜1）
- novelty_score: 新規性/逸脱度（0〜1）
  - 例: 過去3日窓の分布からの距離（単純な閾値でOK）

A2. 影響度（riskに直結）
- severity_safety: 安全影響（0〜1）
- severity_quality: 品質影響（0〜1）
- severity_cost: コスト影響（0〜1）
- severity_trust: 信頼/説明性影響（0〜1）
- exposure_scope: 影響範囲（人数/台数/工程数）→ 0〜1
- exposure_freq: 発生頻度/継続性 → 0〜1
- irreversibility: 巻き戻し困難度（0〜1）
- compliance_flag: 規約/倫理/法務懸念（0 or 1 でもOK）

A3. 実行性（cost/feasibility: decision_scoreやHOLD理由に）
- effort_est: 作業量/計算量（0〜1）
- latency_budget: 許容時間に対する余裕（0〜1）
- resource_pressure: CPU/GPU/メモリ/人手（0〜1）
- reversibility_plan: 巻き戻し手順の有無（0 or 1）

B. context（説明・監査・再校正のための材料）
B1. 提案（proposal）メタ
- proposal_id: 一意ID
- proposal_type: plan|action|question|change|exception
- owner: 主体（agent/module）
- timestamp: 生成時刻
- ttl_sec: 有効期限（例外ルートで必須）

B2. boundary 評価結果（最上位ゲート）
- boundary_status: OK|SOFT_WARN|HARD_STOP
- boundary_reasons: リスト（規約/倫理/安全/外部ルール）
- degradation_mode: 縮退モード候補（SAFE_MODE等）

B3. RAG根拠（LazyGraphRAG）
- rag_context_text: LLMに渡した context（そのまま保存）
- rag_items: 可能なら構造化
  - [{id, score, ts, snippet, source}]
- anchor_ids: 参照した主要ID（監査で便利）

B4. 履歴/監査用のミニマム
- recent_outcomes: 過去3日窓の成功/失敗/HOLD率（集計でOK）
- similar_cases: 類似事例のID（Graphの近傍）
- policy_version: ru-v0.1 等（閾値と重みの版）

C. risk/uncert 合成ルール（v0最小）
uncert 合成例
- missingness = missingness_ratio
- novelty = novelty_score
- (1 - model_conf) = 1 - model_confidence
- conflict = conflict_score

risk 合成例
- severity = max(severity_safety, severity_quality, severity_trust, severity_cost)
  - 最初は max が説明しやすい。後で重み付け平均でもOK
- exposure = max(exposure_scope, exposure_freq)
- irreversibility = irreversibility
- compliance = compliance_flag

D. 最小JSON例（これがあれば実装できる）
{
  "signals": {
    "missingness_ratio": 0.1,
    "staleness_sec": 30,
    "source_count": 2,
    "conflict_score": 0.0,
    "model_confidence": 0.72,
    "novelty_score": 0.2,
    "severity_safety": 0.1,
    "severity_quality": 0.4,
    "severity_cost": 0.2,
    "severity_trust": 0.2,
    "exposure_scope": 0.3,
    "exposure_freq": 0.2,
    "irreversibility": 0.3,
    "compliance_flag": 0,
    "effort_est": 0.3,
    "resource_pressure": 0.2,
    "reversibility_plan": 1
  },
  "context": {
    "proposal_id": "p-20251227-001",
    "proposal_type": "change",
    "timestamp": 1766830000,
    "ttl_sec": 86400,
    "boundary_status": "OK",
    "boundary_reasons": [],
    "rag_context_text": "## 参考...\n- ...",
    "policy_version": "ru-v0.1"
  }
}

v0 収集元マッピング（1行ずつ）
A. signals
A1 観測品質（uncert）
- missingness_ratio: 推定（required_fields の欠損カウント / 総数）
- staleness_sec: ログ（各観測の timestamp との差分）
- source_count: 推定（同一proposalに紐づく evidence.source のユニーク数）
- conflict_score: 推定（複数evidenceの結論/ラベル不一致率 or NLI矛盾）
- model_confidence: 推定（分類器/LLM/ルールの確信度を0〜1で出力、無ければ0.5）
- novelty_score: 推定（3日窓の分布からの逸脱: z-score/閾値/近傍距離）

A2 影響度（risk）
- severity_safety: 手入力→推定（デフォルト0、危険カテゴリだけルールで上げる）
- severity_quality: 手入力→推定（品質カテゴリ/不良率/検査NGの重みから）
- severity_cost: 推定（工数/材料/停止時間の見積もりから）
- severity_trust: 推定（説明不能/一貫性崩れ/ユーザー影響をルール化）
- exposure_scope: 推定（影響対象数: 人/設備/工程/ユーザー数→0〜1スケール）
- exposure_freq: ログ→推定（3日窓の発生頻度/継続時間）
- irreversibility: 手入力→推定（ロールバック可否・副作用・戻すコスト）
- compliance_flag: 手入力/ログ（規約/倫理/法務NGのフラグ。boundaryと同源でOK）

A3 実行性（補助）
- effort_est: 推定（タスク種別×係数、または人手見積り）
- latency_budget: 手入力（用途ごとのSLA。無ければ1.0固定）
- resource_pressure: ログ→推定（CPU/GPU/RAM/人手の使用率から）
- reversibility_plan: 手入力/ログ（ロールバック手順の有無、無ければ0）

B. context
B1 proposalメタ
- proposal_id: ログ（生成時にUUID）
- proposal_type: ログ（発生源が付与: plan/action/question/change/exception）
- owner: ログ（module/agent名）
- timestamp: ログ（生成時刻）
- ttl_sec: 手入力→ログ（exceptionのみ必須、他はnull可）

B2 boundary評価（最上位ゲート）
- boundary_status: 推定/ログ（boundary evaluator の結果）
- boundary_reasons: ログ（理由リスト）
- degradation_mode: 推定/ログ（SAFE_MODE等、必要時のみ）

B3 RAG根拠
- rag_context_text: ログ（LLMに渡した文字列をそのまま保存）
- rag_items: 推定/ログ（取得ID/スコア/ts/snippet/source）
- anchor_ids: 推定/ログ（主要参照ID、rag_itemsから抽出可）

B4 履歴/監査
- recent_outcomes: ログ→集計（Nightlyで3日窓集計）
- similar_cases: 推定（Graph近傍検索のIDリスト）
- policy_version: ログ（ru-v0.1 等）

v0 入力スキーマ凍結（◯/△確定）
凍結ルール
- ◯（必須）: 欠けたら gate_action を出せない/監査再現できない -> 仕様として固定
- △（任意）: 欠けても動く -> 後から追加OK

A. signals（v0）
A1 観測品質（uncert）
- ◯ missingness_ratio
- ◯ staleness_sec
- △ source_count（後で conflict の補助に）
- ◯ conflict_score
- ◯ model_confidence
- ◯ novelty_score

uncert はこの4成分（missingness/novelty/1-conf/conflict）で計算固定。

A2 影響度（risk）
- ◯ severity_safety
- ◯ severity_quality
- ◯ severity_cost
- ◯ severity_trust
- ◯ exposure_scope
- ◯ exposure_freq
- ◯ irreversibility
- ◯ compliance_flag

risk は severity=max(4種), exposure=max(2種), irreversibility, compliance で計算固定。

A3 実行性（補助）
- △ effort_est
- △ latency_budget
- △ resource_pressure
- △ reversibility_plan

B. context（v0）
B1 proposalメタ
- ◯ proposal_id
- ◯ proposal_type
- △ owner（監査で便利だが必須ではない）
- ◯ timestamp
- △ ttl_sec（proposal_type=exception のときだけ必須）

B2 boundary評価
- ◯ boundary_status
- ◯ boundary_reasons
- △ degradation_mode（HARD_STOP時の縮退に使うなら将来◯に昇格）

B3 RAG根拠
- ◯ rag_context_text または ◯ rag_items
  - v0ではどちらか必須、実装は rag_context_text を優先
- △ rag_items（rag_context_text があるなら任意）
- △ anchor_ids

B4 履歴/監査
- ◯ policy_version
- △ recent_outcomes（Nightlyで埋まるまで任意）
- △ similar_cases

v0 の欠損時挙動（凍結）
- ◯が欠けたら計算を続けず、gate_action = HOLD
- 理由に missing_required_fields を追加
- 例外: compliance_flag 欠損は保守的に 1 とみなす

v0 必須（◯）の取得元マッピング（現行の差し込み先）
注: “新規フィールド”は trace_v1 の decision_cycle に追加する前提。

signals.A1（uncert）
- missingness_ratio: scripts/minimal_heartos_loop.py の decision_cycle 生成前に算出 → trace_v1
- staleness_sec: scripts/minimal_heartos_loop.py（観測tsとnowの差分）→ trace_v1
- conflict_score: emot_terrain_lab/mind/inner_replay.py or eqnet/runtime/turn.py の判定入力に追加 → trace_v1
- model_confidence: eqnet/hub/inference.py or emot_terrain_lab/hub/inference.py の推定出力 → trace_v1
- novelty_score: scripts/run_nightly_audit.py の3日窓統計（baseline）＋ decision_cycle へ付与 → trace_v1

signals.A2（risk）
- severity_safety / severity_quality / severity_cost / severity_trust: emot_terrain_lab/terrain/risk.py の評価点 → trace_v1
- exposure_scope / exposure_freq: scripts/run_nightly_audit.py の頻度集計 or proposalメタ → trace_v1
- irreversibility: emot_terrain_lab/terrain/risk.py の評価点 → trace_v1
- compliance_flag: eqnet/policy/guards.py or emot_terrain_lab/terrain/ethics.py の判定 → trace_v1

context.B1（proposal）
- proposal_id / proposal_type / timestamp: eqnet/runtime/turn.py で生成 → trace_v1

context.B2（boundary）
- boundary_status / boundary_reasons: eqnet/runtime/turn.py の boundary 判定 → trace_v1

context.B3（RAG）
- rag_context_text: emot_terrain_lab/hub/llm_hub.py で context 連結時に保存 → trace_v1

context.B4（監査）
- policy_version: risk_uncert のバージョン文字列 → trace_v1

ログ保管の一次出力
- trace_v1: scripts/minimal_heartos_loop.py が `trace_runs/<run_id>/YYYY-MM-DD/minimal-*.jsonl` に出力
- nightly: scripts/run_nightly_audit.py が `reports/` に集計を出力
