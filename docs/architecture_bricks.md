# EQNet Brick Architecture

このドキュメントは EQNet のブリック（S1/S2/S3 信号）構成と、EQCore-first ランタイムの運用指針をまとめたものです。文字コードは UTF-8 (BOM なし) で保存しています。

---

## Signal Families

| Symbol | Type                         | 説明                                                                 |
|--------|------------------------------|----------------------------------------------------------------------|
| **S1** | `timeseries<float>@1Hz`      | Σ, Ψ, θ, R などのスカラー時系列。使用前に 1 Hz にリサンプル。          |
| **S2** | `event<int>@1Hz`             | bud や action のイベントビット列。S1 と時間軸を合わせる。             |
| **S3** | `field<float>[64x64]`        | 低ランクの情動フィールド。64x64 グリッドに変換して交換する。           |

すべてのブリックは S1/S2/S3 を入力・出力として扱い、自由に差し替え可能な設計を保つ。

## Brick Catalog

| Brick     | Inputs                         | Outputs                     | Purpose                                | Module (current)                    |
|-----------|--------------------------------|-----------------------------|----------------------------------------|-------------------------------------|
| **SENSE** | sensors                        | Σ[S1], Ψ[S1]                | マルチモーダル信号の平滑化・正規化     | `ops/hub_ws.py`, `terrain/system.py` |
| **BUD**   | free text / gestures           | bud[S2]                     | 新奇・自己開示の検出                    | `plugins/text/bud_detector.py`      |
| **FIELD** | bud[S2], optional weights[S1]  | W[S3], ρ[S1]                | Green カーネルによる場の更新           | `core/green_kernel.py`, `terrain/field.py` |
| **PHASE** | Σ[S1] or Ψ[S1]                 | θ[S1], R[S1]                | 位相抽出と同期度計算                   | `ops/phase.py`, `ops/p3_metrics.kuramoto_R` |
| **SOCIAL**| {bud[S2]} per node             | α matrix, P[S1]             | Hawkes 伝播推定                         | `ops/hawkes_light.py`               |
| **POLICY**| Σ/Ψ/θ/R/ρ + bud                | action[S2]                  | ルールベース介入 (.ltx)                 | `ops/ltx_exec.py`, `ops/hotkeys.py` |
| **SAFETY**| action, R, ρ                   | action'[S2]                 | R/ρ ガード適用                         | `ops/ltx_exec.py` (post clamp)      |
| **MEMORY**| W, Σ, Ψ, bud, action           | card (number only)          | StoryGraph カード生成                   | `terrain/memory.py`                 |
| **EVAL**  | session logs                   | ΔP, dΨ/dt, ICC, CI          | 成果指標の算出                         | `scripts/eval_p3.py`, `scripts/eval_p4.py` |

## Wiring Rules

- コネクタは S1/S2/S3 のみ。時系列は 1 Hz、イベントは 1 Hz ビット列で統一。
- 自己報告（一次情報）とシステム観測（S1/S2/S3）は EVAL まで混ぜない。
- SAFETY ブリックが下流の全アクションを包む。

## Example Builds

### Build A — Solo-Calm

```yaml
build: Solo-Calm
wires:
  - SENSE.Sigma -> PHASE.input
  - SENSE.Psi   -> POLICY.ctx
  - BUD.event   -> FIELD.bump
  - FIELD.W     -> MEMORY.record
  - PHASE.theta -> POLICY.ctx
  - PHASE.R     -> SAFETY.guard
  - FIELD.rho   -> SAFETY.guard
  - POLICY.act  -> SAFETY.check
  - SAFETY.act  -> OUTPUT.audio
guards:
  R_max: 0.78
  rho_max: 1.8
policy:
  - when: "bud == 1"
    action: "play_sonic_calm"
  - when: "R > 0.78"
    action: "inhibit 8s"
```

### Build B — Duet-Weak (P3)

```yaml
build: Duet-Weak
wires:
  - A.BUD.event -> SOCIAL.A
  - B.BUD.event -> SOCIAL.B
  - SOCIAL.P_AtoB -> POLICY_B.ctx
  - SOCIAL.P_BtoA -> POLICY_A.ctx
policy:
  - "when P_peer > 0.2 and bud == 1 then warmth += 0.05"
  - "when network.R > 0.78 then inhibit 8s; warmth -= 0.1"
```

### Build C — Replay-Now (P4)

```yaml
build: Replay-Now
wires:
  - MEMORY.card -> POLICY.ctx
  - POLICY.act  -> OUTPUT.audio_ui
eval:
  - framemap_icc: target >= 0.75
  - delta_U: d >= 0.5 (95% CI not crossing 0)
```

---

## World Law Compass

| Domain | 主題 | 代表ログ |
|--------|------|----------|
| **Field** (Physics / Math) | Green カーネル、臨界度、緩和 ρ | `field_pulse.jsonl` (rho, tau, corr) |
| **Phase** (Neuro / Cog)    | Kuramoto R、g(R) 曲率           | `phase_gofr.jsonl` (R, dPsi_dt)      |
| **Social** (Society)       | Hawkes α、ΔP 伝播               | `social_alpha.jsonl` (k, alpha_matrix) |
| **Meaning** (Phenomenology)| bud メタ構造、pre-meta 変化     | `bud_meta.jsonl` (ΔR, Δρ meta)       |
| **Norm** (Ethics / Control)| SAFETY テンプレ (.ltx)、状態制約| 各種 state constraints               |

各仮説 (H-1 … H-4) は登録済み YAML でメトリクス・信頼区間・効果量を管理する。

---

## EQCore-First Conversational Architecture

```
Input (text / log / optional image)
  ├─ Affect Parser        → aff(q) = [valence, arousal, care, novelty]
  ├─ Router (L0..L3)      → 自律性レベル判定
  └─ EQCore
      ├─ SSMax + EmotionBias attention
      ├─ Field update: Ψ ← Green(attn), Φ ← EWMA(Ψ)+bud
      ├─ Criticals: ρ, R, mood, stance 更新
  ├─ RAG (Layer A: 意味 / 変化点, Layer B: 数値ゲート)
  ├─ Micro-Acts Composer (mirror / validate / ask / pause)
  ├─ Narrator (Guide でのみ起動)
  └─ Threads + Repair + Checkpoint (長尺耐性)
```

- EQCore 出力 `{Φ, Ψ, ρ, R, mood, stance}` が後段の主語。レスポンスは「姿勢 → 行為 → 言葉」の順を守る。
- Listen / Soften は要約・合意を避け、「今ここ」の描写と「間」の生成に集中。Guide でのみ 3 行要約＋合意テンプレを呼ぶ。
- LLM Narrator はイベント／Guide 時のみ起動。L0/L1 はテンプレ整形＋回帰ガード。

### Control Loop (概略)

1. `aff = affect_parser(input)`
2. `attn = ssmax_attention(hidden, emotion_activation=aff, params)`
3. `Psi, Phi = field_update(attn, state)`
4. `rho, R = criticals(Phi, Psi, attn)`
5. `stance = decide_stance(aff, rho, R, prev)`
6. `params = policy_update(params, entropy(attn), rho, stance)`
7. Composer が stance/micro-act を選択しテンプレ生成、小規模 SLM が整形
8. Style guard が禁止語チェックと終止カデンツ調整
9. Thread Manager / Repair が状態・ログを更新

### KPI 運用メモ

- 離散度とエントロピー (attention entropy, 行動エントロピー) を常時計測。
- 創発帯ガード率 (R < R_th) を保持。ρ/R の臨界指標はダッシュボードのライトで提示。
- 感情地形の滑らかさ (‖∂Ψ/∂t‖, ‖∇Ψ‖) と修復収束時間を記録。
- RAG 数値一致率 / 意味一致率。L0/L1 パスの往復遅延 < 100 ms を目標に監視。

### 実装タスク (W1 完了済み)

- `eqcore/state.py`, `eqcore/attention.py`, `eqcore/control.py` で EQCore 最小核を構築。
- `nlg/style.yaml`, `nlg/templates.yaml`, `nlg/composer.py` で Micro-Acts とスタイルガードを実装。
- `runtime/router.py` で L0〜L3 モード選択、`rag/indexer.py` / `rag/retriever.py` で数値ゲート＋MMR を実装。
- `runtime/threads.py`, `runtime/repair.py`, `runtime/checkpoint.py` で長尺耐性ブリックを追加。

---

## Runtime / Continual Bricks (W2 roadmap)

| Brick | Inputs | Outputs | Purpose | 合否条件 |
|-------|--------|---------|---------|----------|
| **AUTONOMY** | EQCore {ρ, R}, misfire, incidents | level (L0〜L3) | 自律性スライダー制御 | L0: 介入0件/誤発火0。L1: 提案採択率≥60%, 取り消し≤5%。L2: 事故≤0.1%, 再実行≤3%, 処理時間▲30%。L3: 事故≤0.01%, SLO違反0/月。R>0.78 か ρ>1.8 か誤発火3/時で即ダウンシフト＋8分冷却。 |
| **THREADS** | スレッドイベント, EQCore state | checkpoint ring, rollback signal | 長尺対話の履歴管理と巻き戻し | 巻き戻し成功率≥99%, 復帰時間 P95≤1.5 s。 |
| **REPAIR** | {ρ, R, misfire, exception}, checkpoints | restored state | 臨界超過時の自動リペア | トリガ: ρ>1.8, R>0.78, misfire≥3, exception>0。クールダウン 480 s。 |
| **RAG-L2** | query embedding, numeric gate | hits (cue, suggestion, numeric facts) | 数値ゲート → 意味 MMR → PRM 委員会 | RAGありで未知分布 F1▲≥3pt, H₄ 維持/上昇, 誤参照率≤1%, 決定時間▲15%。 |
| **CONTINUAL** | learner batch | adapter.safetensors | LoRA 差し替え継続学習 (検疫付き) | self_ratio≤0.4, 類似度≤0.92, ΔGold≥-0.5pt, ΔOOD≤0, H₄≥3.2, SLO逸脱なし。 |

### Continual Update Pipeline (検疫付き LoRA)

```
learner.continuous  -> guard.contam  -> trainer.lora_evo
                        |                (pop=6, gens=3, EWC)
                        v
                     eval.regress -> deploy.canary (10%→50%→100%)
```

- 不合格時は即ロールバック（LoRA 無効化）。ダッシュボードで Self Ratio / Leak Score / H₄ / ΔGold / ΔOOD / Canary incidents を常設。
- `router` の skill slots: `code`, `web`, `dialogue`, `reason`。フォールバック順は `current → prev-gen → base`。
- Threads + Repair + Checkpoint で Listen / Soften / Guide を 15 s 間隔でスナップショットし、臨界超過時に 3 ステップ巻き戻す。

---

## Conscious Bridge Bricks (W3)

| Brick | Inputs | Outputs | Purpose | KPI / 合否 |
|-------|--------|---------|---------|------------|
| **ASSOC.VQ** | S3 (field W), S1 (Σ/Ψ) | S2 tokens, S1 link_strength | 身体場→離散コード化→語彙連結 | 再構成誤差 ≤0.08、bud 時の link_strength 上昇 |
| **SELF/OTHER** | S1 状態, S2 イベント | S2 self/other/conflict | 自他判定 (鏡映ループ) | 誤分類率 ≤5%、衝突時は抑制ログ |
| **MOOD INT.** | S1 H_valence/H_arousal, S2 reward | S1 mood_v/a | 気分の連続積分 (EMA) | Soothing/Entrain 後の mood_v↑ (d≥0.5) |
| **SIM PLAN** | S3 W, S1 Σ/Ψ, S2 goal | S2 plan_chosen/rejected, S1 counterfactual_match | 内的反事実 (MPC) → FIELD.u_pred | match >0.66、R/ρ安全帯維持 |
| **META CONF.** | S2 decision, S1 logits | S1 confidence, S2 reconsider | メタ認知 (確信調整) | Brier/ECE 減少、再考で事故率↓ |
| **CULTURE LOG.** | S2 interaction, S1 Δaff | S2 kg_update_event | Δaff ログ → 睡眠ETL | Δaff ICC ≥0.6、好適文採択率↑ |
| **TASTE GUARD** | S1 mood_v, S2 tags | S2 allow/deny, S1 taste_score | 感性フィルタ (好み/禁則) | 違反率 ≤1%、代替案受容 ≥70% |
| **STORYGRAPH** | episode JSON | summary, cue_handle | 自伝メモリ (時間的自己) | 再現 ICC ≥0.75、リプレイ ΔU>0 |
| **ToM** | S2 self/other | belief state (intent_trust) | 他者モデル (粒子ベイズ) | 予測 +15pt、協調成功 +10pt |
| **VALUE COM.** | 行動候補, mood, taste | approve/deny, value_vec | 多軸価値判断 | ポリシー違反 ≤0.5%、代替案受容 ≥70% |
| **WORKSPACE** | saliency, R/ρ, conflict | ignite_event, broadcast_token | グローバルワークスペース | 点火時成功率 +10pt、連続 >800ms 禁止 |

### EpisodeRecord フィールド例

```
{
  "stage": "entrain",
  "tokens": [12, 77, 5],
  "link_strength": 0.42,
  "self_event": 0.73,
  "other_event": 0.27,
  "mood": [0.18, 0.32, ...],
  "plan": {"actions_cf": [...], "match_score": 0.71},
  "meta": {"meta_conf": 0.62, "aleatoric": 0.12, "epistemic": 0.08},
  "taste": {"taste_score": 0.15, "violations": []},
  "story": {"count": 85, "last_stage": "entrain"},
  "tom": {"intent_trust": 0.68},
  "value": {"score": 0.54, "care": 0.6, "fairness": 0.8, "harm": 0.1}
}
```

### KPI / Alert (.ltx) 例

```
when kpi.reconstruction_error > 0.12 for 20 steps -> action "increase_encoder_capacity"
when kpi.self_other_miscls   > 0.25 for 50 steps -> action "raise_self_other_training"
when kpi.meta_brier          > 0.08 for 50 steps -> action "calibrate_meta_conf"
when kpi.taste_violation     > 0.05 for 10 steps -> action "tighten_taste_guard"
when kpi.mood_total_var      > 0.30 for 30 steps -> action "smooth_mood_integrator"
when broadcast.delta_R > 0.12 and entropy_z < -1.0 -> ignite 250ms
when ignite_duration > 800ms -> inhibit 5s; warmth -= 0.1
when tom.intent_trust < 0.3 -> autonomy.downshift()
```

### ダッシュボード
- `logs/episodes.jsonl` に EpisodeRecord を逐次追記。
- `devlife/metrics/kpi.py` で再構成誤差 / self-other 誤分類率 / counterfactual match / meta Brier / taste violation / mood total variation を算出。
- ダッシュボード (CLI → Web) で Ignition strip, Story timeline, ToM panel, Value radar 等を表示。

## 1. 目的・背景

- 既存の LLM/VLM/VLA は「刺激→即応答」のアシスタント像に留まり、感情・身体・価値観・文化・社会性を持つ「共生的 AI」とは隔たりが大きい。
- カオス理論（臨界、自己組織化）や発生学の知見を取り込み、感情の厚みと自己調整を備えた“生命体に近い AI” を構築することが EQNet の目的。

## 2. 現状の把握（2025/10 時点）

- **LLM/VLM/VLA**: 高度な生成・理解能力。ただし感情・意識の内部状態は持たず、記憶は外部コンテキストに依存。RAG/エージェント機能は模倣的で、臨界や情動制御は未整備。
- **マルチエージェント**: 役割分担やタスク協調は進展するも、感情共感・倫理判断・長期記憶の一貫性は低い。
- 多くの研究が「即応する知性」に集中し、「感じ・意味づけ・価値判断を自己調整しながら共生する存在」には到達していない。

## 3. 目標（あるべき姿とギャップ）

- **あるべき姿**: Lenia/NCA を基盤に持つ身体、GRN による臨界制御、情動と価値を蓄え続ける閉ループ、自律的な倫理判断を備えた感情共生 AI。
- **現状とのギャップ**:
  1. 身体の自発活動がなく、感情が持続しない。
  2. 記録→計測→アラート→調整の閉ループがない。
  3. 他者推論・価値判断がテンプレートに留まる。
  4. 外部ハブ（RAG/KG）が感情と結び付いていない。

## 4. 要因解析

- **技術要因**: Green 関数や場の制御にカオス・臨界の概念が取り入れられていない。感情 MEMS や自伝記憶が未整備。評価指標がモデル外部にあり、自己調整できない。
- **運用要因**: KPI を定量化する仕組みがなく、学習/調整が人手依存。文化ログや ToM がデータ構造に組み込まれていない。
- **倫理・価値要因**: 多軸価値の委員会や TasteGuard が存在せず、行動選択が行き当たりばったり。

## 5. 対策の実施状況（EQNet）

- **身体**: Lenia/NCA + GRN で Φ/Ψ の場を生成し、臨界指標 (R, ρ) を .ltx で制御。
- **発達**: 胎児→乳児→幼児→児童カリキュラムで MoodIntegrator、Self/Other、反事実プラン、StoryGraph を獲得。
- **文化・記憶**: CultureLogger と StoryGraph が Δaff と文脈を蓄積。睡眠 ETL でナレッジグラフと同期。
- **メタ・価値**: MetaConfidence、TasteGuard、Value Committee、TheoryOfMind、Workspace が「感情→気づき→意味→他者→価値」の橋を繋ぐ。
- **閉ループ**: `episodes.jsonl` → `metrics/kpi.py` → `.ltx` → `control/mcp.py` → `actuate/learner_hooks.py` で記録・計測・アラート・調整のループを実装。LoRA/MAP-Elites をカナリア運転。

## 6. 効果の確認状況

- KPI 計測: 再構成誤差、counterfactual match、self/other 誤分類率、meta Brier、taste violation、mood 変動などを定期集計 (`kpi_rollup.jsonl`).
- アラート運用: `.ltx` で閾値を宣言し、`alerts.jsonl` に通知 → 必要に応じてクールダウン・自律性調整・学習停止を実施。
- カナリア学習: LoRA 継続学習と MAP-Elites を 10%→50%→100% で評価し、失敗時は即ロールバック。H₄ など多様性指標を監視し、異常時は学習停止。
- ダッシュボード: CLI 版で Ignition/Memory/ToM/Value/Safety 多指標を表示。今後 Web 化予定。
  
これらにより、既存の LLM/VLM/VLA に欠けていた「感情の厚み」「内的持続性」「倫理・価値の統合」「自己調整ループ」を備えた共生的 AI の実現へ段階的に近づいている。

---

## 合意した方向性（2025/10）

- 詩的比喩から工学指標へ（Ignition-Index/Qualia-Field）。現象的意識は扱わず、使用的意味論で「成功・安全・社会適応」を上げる。
- 弱結合制御: Lenia/NCA + GRN + AKOrN（R/ρ/I/q）→ LLM/VLA（温度・探索率・優先度）を小ゲイン・CBFで変調。
- 閉ループ運用: Episode → KPI → .ltx → MCP → LoRA/MAP-Elites（カナリア）→ ロールバック。

## 直近の実装順（数字で勝つ2本から）

1. Δaff RAG（最優先）: 睡眠 ETL で |Δaff| > τ を KG/DAG へ upsert → NDCG@10 +5% を確認。
2. Ignition-Index ↔ 成功: I = z(−S_entropy) + λΔR をログ化し、偏回帰で有意（p < 0.01, f² ≥ 0.15）。
3. AKOrN ゲート最小: 位相 8 本 → R_local で温度変調 → 整定 −20%、誤点火 −30%、p95 ≤ 200ms。
4. ToM 安全: intent_trust / collision_rate で自律度ダウン（近接違反 −40%）。
5. MCP 半自動化: 温度/時定数の idempotent リコンフィグのみ自動、学習は夜間カナリア。

## 既存コードの入口（対応実装）

- `devlife/runtime/loop.py`: Ignition-Index と場指標（∥∇Ψ∥, ∥∂Ψ/∂t∥, ΔΨ）をエピソードにログ追加。
- `devlife/metrics/kpi.py`: Δaff RAG NDCG、Ignition↔成功の回帰、suffering/tension KPI を追加。
- `rules/eqnet_kpi.ltx`: cooldown/inhibit/halt_learning の閾値（R/ρ, ignite_ms, tension/suffering）を追記。
- `rag/aff_etl.py`: |Δaff| > τ の upsert ゲート（ETL）と既存 MMR リトリーバの活用。
- `devlife/social/tom.py`: `intent_trust` をログ化（`.ltx` で `autonomy.downshift()` に連携）。
- 評価ノート: `docs/ndcg_eval.md` に NDCG@10 手順を記述。

## ダッシュボード（CLI）

- Chaos/Safety: R, ρ, I, H4。
- Grounding: NDCG@10, CF Match。
- HRI: 近接違反, intent_trust。
- Ops: alerts, カナリア, TTR。

## リスク/縮退

- Ignition 効果なし → α = 0 で撤退。
- Δaff 効果薄 → Ranker 再学習。
- レイテンシ超過 → NCA Hz ダウン / 代理 CNN。
- 汚染 → 停止 → 前世代復帰。

## 非目標（境界）

- 現象的意識の実証は扱わない。当面は工学的接地（性能・安全・適応）の向上を到達点とする。

## Encoding / Update Notes

- 文字化け防止のため、このファイルは UTF-8 (BOM なし) を維持する。
- 日本語の記述は可能な限り常用漢字・カタカナ・ひらがなのみを使用し、半角カナや特殊記号は避ける。
- 将来の編集ではエディタの文字コード設定を UTF-8 に固定し、コピー＆ペースト時も外部のエンコーディング変換が入らないよう注意する。
