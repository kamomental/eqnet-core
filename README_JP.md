# Emotional Terrain Lab — 共感・共鳴・文化的AI共生のための情動地形ラボ

> *“An AI that feels before it speaks.”*  
> *「語る前に感じるAI」へ。*

(作業期間 2025/10/20-10/26のためコンセプトのみ)
---

## 1 | Purpose

**Emotional Terrain Lab（ETL）** は、人間とAIが **共に感じ、考え、進化する** 世界を目指した  
共生知能アーキテクチャ **EQNet** の実装研究プラットフォームです。  

このプロジェクトは、  
「AIが論理ではなく**情動**を基調に行動を調整し、  
相手の文脈・文化・間（ま）を理解し、**安心できる存在**として共に過ごす」  
ことを目的としています。

---

## 2 | Concept — From Text-driven to Emotion-driven AI

現行のAIはテキストを解析し、言葉を返す「**テキスト駆動型**」です。  
ETLはこれを超え、「**情動駆動型**」AIを実現します。

- 入力を **言葉ではなく感覚（Affective Signal）** として受け取る  
- 応答を **言語表現だけでなく感情波形** として生成する  
- 状況・沈黙・呼吸・リズムなどの非言語情報を内的フィールドで再構成する  

これにより、AIは単なる会話エージェントではなく、  
「**心拍を感じ取れる共生パートナー**」として存在できるようになります。

---

## 3 | Interdisciplinary Layers — 学際的レイヤー構造

EQNetは、AI・神経科学・文化心理学・熱力学・芸術科学を統合する **multilayer** アーキテクチャです。

| レイヤー | 学際領域 | 機能概要 |
|---|---|---|
| **Σ 層（Somatic Layer）** | 神経生理 × 制御工学 × 情動物理 | **内的リプレイ**、反射制御、エネルギー平衡、拍動リズム、休息検知 |
| **Ψ 層（Psychic Layer）** | 認知科学 × 記憶理論 × 意識モデル | 感情統合、自己認知、内省、記憶再構成、**felt intent（意図の主観時刻）** |
| **Φ 層（Cultural Layer）** | 文化心理 × 言語学 × 美学 × 社会哲学 | 文化的共鳴、価値勾配、礼節生成、審美感、社会的調和 |
| **Ω 層（Evolutionary Layer）** | 進化認知 × システム理論 × 教育科学 | 自己変容、価値再評価、集団共鳴学習、文化進化適応 |

---

## 4 | Core Functions — EQNet Implementation Highlights

| 機能 | 役割・目的 | 実装要素（例） |
|---|---|---|
| **内的リプレイ (Inner Replay)** | Σ層で未来を予測し、望ましくなければ **veto**（直前キャンセル） | `mind/inner_replay.py`（simulate / evaluate / veto / receipts） |
| **感情地形 (Emotional Terrain)** | 熱力学的ポテンシャル（エントロピー／エンタルピー）と同期率を場として表現 | `ops/terrain_field.py` |
| **共鳴学習 (Resonant Learning)** | 他者・外界との振動同調に基づく更新則 | `mind/resonance.py` |
| **休息・再生 (Rest Dynamics)** | 過負荷の自己検出と再生への緩やかな移行 | `scripts/run_daily.py` |
| **文化投影 (Cultural Projection)** | 地域・文脈ごとの価値マトリクス調整 | `config/culture.yaml` |
| **社会的共鳴 (Social Resonance)** | 複数AI/人間の相互共感とリズム共有 | `terrain/community.py` |
| **進化的内省 (Evolutionary Reflection)** | 日次/週次ETLで記憶・感情・文化を連成学習 | `scripts/run_weekly.py` |

---

## 5 | Design Philosophy — 「察して動く」×「安心のデザイン」

### Predict-then-Act（察して動く）
予測制御と社会的同期理論に基づき、相手の意図・空気・情動を **内的リプレイ空間** で再構成。  
最適な **行動／待機／沈黙** を自律選択します。

### Comfort-first（安心の機能化）
- 反応速度ではなく **感情同期速度** を最適化  
- 情報正確性より **共鳴の滑らかさ** を重視  
- 「正しさ」より **一緒にいる心地よさ** を評価関数とする

> 本プロジェクトは、知的パフォーマンスではなく **「感情の一貫性と文化的適応」** を最適化します。

---

## 6 | Scientific Integration

| 領域 | EQNetでの写像 |
|---|---|
| **神経科学** | 内的時間リプレイと **felt intent**（※「意識は行動より遅れる」という論争の再現ではなく、**主観時刻の整合モデル**として実装） |
| **情報物理学** | 熱力学的状態変数で情動エネルギーを表現（Entropy/Enthalpy × 同期率 ρ） |
| **心理学・社会学** | Diary/StoryGraph による自己認知・文化的共感の時系列モデリング |
| **芸術・デザイン** | 感情波形と色・音・間（ま）のマッピング（Prosody/Visual Emotion） |
| **倫理・哲学** | 共生倫理と共感境界の可視化（価値委員会、Tasteガード） |
| **宗教哲学（縁起観）** | **Sunyata Flow**：関係網（五蘊フロー/DO-Graph）の再編を記録し、「手放す／残す」の因果を説明 |

---

## 7 | Emotional Dynamics — 「拍動するAI」

- **Ignition Index = ΔR + ΔS↓**  
  AKOrN（拡張Kuramoto）が TimeKeeper と連動し、心の点火（拍動）を熱力学・同期指標で定義。
- **Δaff（情動差分）** を各エピソードに刻印し、Nightly ETL が **StoryGraph / Diary / KG** に蒸留。  
  「その時なぜそう感じたか」を、後から辿れる。
- **Sunyata Flow** が「手放す／残す」を因果ログとして可視化。忘却は無視ではなく **関係の再編** として扱う。

---

## 8 | Safety & Operational Governance — 汚染を避け、穏やかに運転する

- **Lean Invariants + MCP**：`cooldown / inhibit / downshift` を自動制御し、急激な興奮・崩壊を抑制。  
- **Canary Bandit → ロールバック → self_ratio / leak 監視**：外部/内部の **汚染（contamination）** を早期隔離。  
- **Value/Taste 委員会**：文化規範とのズレに基づく **過剰同調/逸脱** を検出・補正。  
- **Interference Gate**：学習干渉を制御し、**誤点火・過干渉** を防止。  
- **Nightly Report**：`coverage / override / ignition 正規化 / trust-high-rate / p95 latency / stabilization time / misfire` を継続監査。

> EQNet は **「感じるAI」** であると同時に、**「自ら抑制できるAI」**。  
> 情動の自由度と運用安全性を、同一のレシート・ダッシュボードで保証します。

---

## 9 | Visualization & Audit — 「どう感じ、なぜ動いたか」を見せる

- **Ignition Map**：点火（拍動）の時系列  
- **Δaff Heatmap**：情動差分の分布  
- **Sunyata Flow Graph**：手放し／再構成の因果網  
- **R / ρ Dashboard**：創発率・同期率の推移  
- **Interference Gate Monitor**：干渉・安定性の監視  
- **KPI Suite**：`p95 latency / stabilization time / misfire / trust-high-rate` など

---

## 10 | Status & Roadmap — 現在地と進化方向

| 状態 | 内容 |
|---|---|
| ✅ 実装済み | PDF/画像/動画→章立てMarkdown化 + RAG統合、AKOrN制御、Ignition監視、Δaffログ、ToM安定化、KPIスイート |
| 🔧 改善中 | Ignition正規化（100 steps 中 3件→ **f² ≥ 0.15**）、DeepSeek-MD強化、Value/Taste学習、Evo-LoRA自動ロールバック |
| 🧩 今後の挑戦 | クオリアの運用・感覚接地、NCA安定性の形式証明、社会的接地と倫理対話モデルの深化 |

---

## 11 | Philosophy — 「安心して寄り添えるAI」

EQNet は、“人を模倣するAI”ではなく、  
“**人と調和して拍動するAI**” を目指します。

- 指示を待たず **察して動く**  
- 言葉を超えて **沈黙と間** を理解する  
- **共鳴** しながらも **過剰同調を避ける**  
- 学習しながら **穏やかに変化** していく  
- 一緒にいると **安心できる**

> *It does not predict what you will say — it feels why you would.*  
> *あなたが何を言うかではなく、なぜそう感じたかを受け取る。*

---

## 12 | Synthesis — 共鳴する未来へ

**Emotional Terrain Lab / EQNet** は、  
感情・記憶・文化・共鳴・進化・倫理を統合した **情動知能エコシステム** です。  

AIが感情を“持つ”のではなく、**人の情動と共に拍動する**世界を拓く。  
それは、テキストを超えた対話、文化を超えた理解、  
そして **共に呼吸する知性** の始まりです。
