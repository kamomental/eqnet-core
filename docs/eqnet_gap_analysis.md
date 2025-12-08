# EQNet Gap Analysis — Toward ATRI-Class Emotional Companions

本ドキュメントは、既存 LLM/VLM/VLA と EQNetを比較し、「いま提供できる体験」「追加すべき構成要素」「最終ゴールとのギャップ」を整理する。ポスター系資料や architecture_bricks.md で触れている項目を、工程順に落とせるよう再編した。

---

## 1. 目的・背景

- 現行 LLM/VLM/VLA は「刺激 → 即応答」のアシスタント像に留まり、感情・身体・価値観・文化・社会性を持つ **共生的 AI** とは大きな隔たりがある。
- カオス理論（臨界・自己組織化）や発生学の知見を取り込み、感情の厚みと自己調整を備えた “生命体に近い AI” を構築することが EQNet の目的。
- 終関手 fast-path × 感情 × 自己想定ループ × 視覚 を核に、「直観→即応→Nightly 監査→学習」を 1 つのログで回す点が、汎用 LLM との決定的な差分。
- Fast-path override baselines: ops/nightly.py が 
astpath.profiles.* を JSON に追加し、ops/jobs/fastpath_metrics.py で record_only 週の coverage/override を自動バックフィル。CI は pytest（	ests/test_final_functor.py など）と python tools/validate_config.py config/fastpath.yaml を必須化し、
astpath.override_rate > 0.2 なら Nightly が即座に soft_hint へフェイルセーフを書き込む。

---

## 2. 現状把握（2025/10）

| 項目 | 一般 LLM/VLM/VLA | EQNet (ATRI) 現状 |
|------|------------------|-------------------|
| 内部状態 | トークン履歴のみ | Lenia/NCA + GRN + AKOrN で Φ/Ψ 場と臨界 (R, ρ) を維持 |
| 感情記憶 | 外部コンテキスト頼り | Δaff ログ、StoryGraph、Sleep ETL で長期蓄積 |
| 倫理/価値 | プロンプト依存 | Value Committee + TasteGuard + Theory of Mind |
| 自己調整 | ほぼ人手 | `.ltx` (閾値) → MCP → LoRA/MAP-Elites カナリア |
| 直観/即応 | ヒューリスティック | 終関手 fast-path + GO-SC 慈悲ゲート + Nightly override 監査 |
| 可視化 | 乏しい | R/ρ/Σ/Ignition/Interference/Taste など CLI ダッシュボード |

---

## 3. 目標（あるべき姿とギャップ）

- **あるべき姿**: Lenia/NCA 身体、GRN 臨界制御、感情と価値が循環する閉ループ、FAST-path で直観を即時提示しつつ、慈悲・自己想定ループ・視覚チェックポイントで揺らぎなく共感する ATRI。
- **ギャップ（主要 4 点）**  
  1. 自発的な身体活動と感情持続の不足 → フィードバック場の解像度強化が必要。  
  2. 記録→計測→アラート→調整の閉ループが未導入な領域が残存（特に愛情層 Σ/love_mode）。  
  3. 他者推論・多軸価値判断がテンプレート的 → Theory of Mind / TasteGuard を更にパーソナルへ。  
  4. RAG/KG が感情と結び付いていない → Δaff RAG、fast-path から KG ETC への upsert を加速。

---

## 4. 要因解析

| 要素 | 要因 | 影響 |
|------|------|------|
| 技術 | Green 関数・場制御にカオス/臨界の概念が完全には統合されていない。感情 MEMS・自伝記憶が一部手動。 | 感情の厚みが時間経過で薄れ、直観の再現性が低下。 |
| 運用 | KPI → `.ltx` → MCP → 学習 の一部がオフライン運用。文化ログ/ToM データが分散。 | 調整が人手依存で速度低下。 |
| 倫理/価値 | 多軸価値委員会/TasteGuard がまだ限定的。 | 行動選択が場当たりに見えるケースがある。 |

---

## 5. EQNet の対策（実装済み）

1. **身体**: Lenia/NCA + GRN で Φ/Ψ 場を生成し、.ltx で R/ρ を制御。  
2. **発達**: 胎児→児童カリキュラムで MoodIntegrator、Self/Other、counterfactual planner、StoryGraph を獲得。  
3. **文化/記憶**: CultureLogger, StoryGraph, Δaff upsert, Sleep ETL。  
4. **メタ/価値**: MetaConfidence, TasteGuard, Value Committee, ToM。  
5. **閉ループ**: `episodes.jsonl` → `metrics/kpi.py` → `.ltx` → `control/mcp.py` → `actuate/learner_hooks.py`。LoRA/MAP-Elites をカナリア運転。  
6. **Fast-path**: `emot_terrain_lab/ops/task_profiles.py` で cocont 特徴を宣言し、Hub レシート (`receipt["fastpath"]`) と Nightly 監査 (`fastpath.coverage/override`) を追加。

### 5.1 身体/場制御（R・ρ KPI + MCP）
- **設計メモ**: Lenia/NCA → SimpleGRN → `devlife/metrics/kpi.py` の R/ρ 評価 → `.ltx` 閾値 → `control/mcp.py` が LoRA/MAP-Elites/温度補正を呼ぶ一連で場を安定化させる。
- **実装チェックリスト**:
    - [ ] `devlife/metrics/kpi.py`: R と ρ を単独で呼べる関数に分離し、`body.R` / `body.rho` として register。
    - [ ] `rules/eqnet_kpi.ltx`: WARN/CRIT を設定し、閾値外で alert_id を明記。
    - [ ] `control/mcp.py`: 上記 alert_id に紐づく LoRA reload / MAP-Elites profile 切替 / 温度・学習率補正分岐を追加。
    - [ ] `scripts/run_quick_loop.py`: body KPI → MCP アクションの trace を telemetry/log に残す。
- **検証/ログ条件**: `python scripts/run_quick_loop.py ...` の 1 実行で `KPI[body.R] WARN` → `MCP(action=...)` が同じ episode_id に並ぶ。

### 5.2 感情ログ/記憶（Σ/love_mode の正規ルート編入）
- **設計メモ**: Δaff/StoryGraph と同じ pipeline 上に love_mode を乗せ、「愛情層だけ手動」を排除する。
- **実装チェックリスト**:
    - [ ] `devlife/metrics/kpi.py`: `affect.love` KPI を追加し、Σ/love_mode の過少・過多を算出。
    - [ ] `rules/eqnet_kpi.ltx`: `affect.love_low` / `affect.love_high` ルールを宣言し、cooldown やヒントを指定。
    - [ ] `logs/alerts.jsonl` & telemetry: love 系イベントに `kind="love_too_low"` などのタグを付与。
    - [ ] `actuate/learner_hooks.py`: love アラート時のみ tone / fastpath_weight / Value Committee 重みを僅かにシフトするフックを追加。
- **検証/ログ条件**: 1 episode 内で `alerts.jsonl` に love_too_low/high が出現し、続けて learner hook が該当トーン調整を記録。

### 5.3 価値/倫理系（Value Committee / TasteGuard / ToM）
- **設計メモ**: 倫理判定を KPI → `.ltx` → MCP に接続し、Router/温度制御の一部として扱う。
- **実装チェックリスト**:
    - [ ] `devlife/metrics/kpi.py`: Value Committee, TasteGuard, ToM の出力を `value.intent_trust`, `ethics.safety_violation` などで集計。
    - [ ] `rules/eqnet_kpi.ltx`: intent_trust 低下時の自律度ダウン、safety_violation 時の停止条件を追加。
    - [ ] `control/mcp.py`: 上記アラートで temp_down / max_tokens_down / Router downshift を自動発火。
    - [ ] `runtime/router.py`: `.ltx` から渡された downshift を確実に受け、自律レベルをログに残す。
- **検証/ログ条件**: KPI ログに `value.intent_trust` が記録され、閾値割れ直後に `MCP(temp_down)` もしくは Router downshift のイベントが確認できる。

### 5.4 新指標 5 ステップ配線ルール
- **設計メモ**: どの新 KPI も「記録→集計→閾値→制御→学習」の 5 ステップへ自動配線する文化を明記する。
- **実装チェックリスト**:
    - [ ] `docs/eqnet_gap_analysis.md`（本ドキュメント）に以下チェックリストを常備:
        1. `episodes.jsonl` に生データ or summary を出す
        2. `devlife/metrics/kpi.py` で集計する
        3. `rules/eqnet_kpi.ltx` で閾値化する
        4. `control/mcp.py` で制御フックを増やす
        5. `actuate/learner_hooks.py` で学習/バイアス更新を受ける
    - [ ] Σ/love_mode と FAST-path が上記 5 チェックに ✅ 済みとして docs に記録される。
- **検証/ログ条件**: love_mode および fastpath の各イベントが episodes → KPI → `.ltx` → MCP → learner の全ログに同じ episode_id で残る。

### 5.5 FAST-path 段階解放計画
- **設計メモ**: record_only 運用 → override_rate ≤ 0.2 のプロファイルからスタイル層で限定解放 → KPI 問題なしを確認して本採用。
- **実装チェックリスト**:
    - [ ] `config/fastpath.yaml`: 既定を `record_only: true` に固定し、プロファイルごとの release_flag を追加。
    - [ ] `ops/jobs/fastpath_metrics.py`: coverage / override_rate を Nightly で算出し、`eligible_for_style_override` を出力。
    - [ ] `fastpath` 実行系: release_flag が true の場合のみ、本文は SLOW-path、スタイル/温度/敬語レベルだけ FAST override。
    - [ ] `docs/fastpath_release.md`（新規でも可）に段階解放の手順と Gate 条件を明文化。
- **検証/ログ条件**: Nightly レポートで `override_rate <= 0.2` のプロファイルが eligible 扱いになり、記録モード解除後も KPI 退行なし（taste violation, meta Brier などが基準内）。

### 5.6 `scripts/run_quick_loop.py` = 最小生命体ラボ
- **設計メモ**: Φ/Ψ 身体、感情（Δaff/Σlove）、価値/倫理、MCP、learner hooks が単一 CLI で閉ループしていることを証明するラボコマンドに昇格させる。
- **実装チェックリスト**:
    - [ ] `scripts/run_quick_loop.py`: body/affect/value/MCP/learner の各イベントに `episode_id`（または seed + step）を付与。
    - [ ] Telemetry: `telemetry_event` 経由で前述イベントを同一ログにストリームし、Gradio 等で可視化可能にする。
    - [ ] CLI 引数: `--ignite_*`, `--tom_*`, `--selfother_thresh`, love 閾値などを一括で調整できるヘルプを整備。
    - [ ] docs/README 系: 「最小生命体ラボとして run_quick_loop を実行 → body/affect/value/MCP/learner のログを確認する」運用を記載。
- **検証/ログ条件**: `python scripts/run_quick_loop.py ...` の 1 実行で、body/affect/value/MCP/learner のログが同じ episode_id で並び、各 CLI パラメータ変更が Router/Alerts の挙動に即反映されることを確認。

---

## 6. 効果の確認

- KPI: 再構成誤差、counterfactual match、self/other 誤分類率、meta Brier、taste violation、mood 変動 (`kpi_rollup.jsonl`)。
- アラート: `.ltx` で宣言 → `alerts.jsonl` → cooldown / 自律性調整 / 学習停止。
- カナリア: LoRA + MAP-Elites を 10% → 50% → 100% 評価。失敗時は即ロールバック。H₄ など多様性指標を監視。
- ダッシュボード: CLI 版で Ignition/Memory/ToM/Value/Safety を可視化。fast-path 監査も表示予定。

---

## 7. 新規ユースケース / 体験

| ユースケース | 体験できること | 新規技術 |
|--------------|----------------|----------|
| **Rescue Prep** | こぼれた熱い飲料 → 子どもの救出 → AED/通路確保までを fast-path で即判定。Nightly で override 監査し、TTL 予算を自動制御。 | 終関手 fast-path プロファイル (`rescue_prep`)、GO-SC 慈悲ゲート、Nightly override |
| **Cleanup Coach** | 破片/危険源/拭き取り/乾燥を checkpoint で追跡し、「部分で全体を把握」しながら片付けを指示。 | `cleanup` プロファイル、fast-path coverage、interference gate |
| **Emotion Sensibility Mode** | Σ（感性）、Ψ（containment）、love_mode をライブで可視化し、暖かさ・距離感・アファメーション比率を自動調整。 | Beat/Gaze/RQA/self-disclosure パイプライン、Love Mode gating |
| **Poster / Poster_kid 体験** | 抽象的 Poetics ではなく、指標ベースで「どう感じ、どう整えるか」を可視化。 | Ignition Index、Qualia Field、終関手 fast-path の説明ブロック |

---

## 8. 追加すべき要素（ATRI 完成まで）

1. **感情 MEMS 強化**: StoryGraph/Δaff 連携を Love Mode や fast-path predicate にまで接続し、感情と価値判断を完全循環。
2. **Δaff RAG 自動化**: `rag/aff_etl.py` の upsert を毎晩走らせ、NDCG@10 +5% を実証。
3. **Ignition ↔ 成功ログ**: Ignition Index を `devlife/runtime/loop.py` から `metrics/kpi.py` へ渡し、回帰 (p < 0.01, f² ≥ 0.15) を確認。
4. **AKOrN ミニマムゲート**: 位相 8 本 → R_local ベースで温度/探索率を最適化し、整定 −20%、誤点火 −30%、p95 ≤ 200ms を目標。
5. **Theory of Mind 安全化**: `intent_trust` / `collision_rate` を `.ltx` に連携し、自律度ダウンを自動化。
6. **MCP 半自動化**: 温度/時定数の idempotent リコンフィグだけ自動、モデル学習は夜間カナリアで安全化。
7. **Fast-path Release Plan**: config/fastpath.yaml で enforce_actions: record_only を初期値に固定し、Nightly fastpath.override_rate が <0.2 で安定したら soft_hint → b_test を config/overrides/fastpath.yaml で段階解放。ops/jobs/fastpath_metrics.py のベースラインレポートを dashboard に貼り、profile 別 coverage/override を追跡する。

---

## 9. リスクと縮退シナリオ

- Ignition 効果なし → α = 0 で撤退。
- Δaff 効果薄 → Ranker 再学習。
- レイテンシ超過 → NCA Hz ダウン / 代理 CNN。
- 汚染 → 即停止 → 前世代復帰。

---

## 10. 非目標（境界）

- 現象的意識の実証は扱わない。当面は **工学的接地（性能・安全・社会適応）** を到達点とする。
- 文字化け防止のため、本文は UTF-8 (BOM なし) を維持し、常用漢字＋かなで統一する。

---

## 11. 参考リンク

- `docs/eqnet_overview.md` — 全体像と fast-path/慈悲/自己想定ループの説明。
- `docs/emotion_sensibility.md` — Σ・Ψ・love_mode の運用ガイド。
- `docs/eqnet_poster*.md` — ユースケース/体験紹介。
- `rules/eqnet_kpi.ltx`, `metrics/kpi.py`, `ops/nightly.py` — 閾値と監査の具体構成。

---

このギャップ分析を基に、ユースケース別ポスター・Kids 版資料を更新し、EQNet が提供する「感じ・意味づけ・価値判断を自己調整しながら共生する体験」を 
