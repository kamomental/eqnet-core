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
7. **Fast-path Release Plan**: config/fastpath.yaml で enforce_actions: record_only を初期値に固定し、Nightly astpath.override_rate が <0.2 で安定したら soft_hint → b_test を config/overrides/fastpath.yaml で段階解放。ops/jobs/fastpath_metrics.py のベースラインレポートを dashboard に貼り、profile 別 coverage/override を追跡する。

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
