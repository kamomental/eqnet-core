# EQNet Emotional Engine Hub — Poster Overview

EQNet は「感じて（Feel）、合わせて（Sync）、大切にする（Care）」を数式とログで運転するヒューマン×AIスタックです。最新の Sunyata / Self / Candor ループを含め、いま提供できる要素を一枚にまとめました。

---

## 0. Vision
- 感情・注意・身体リズムをリアルタイムに可視化し、**人の前で安心して使える AI** を提供する。
- 文化や文体の違いを圧縮しつつ、**「なぜそう答えたか」** をレシートで説明できることを大原則にする。

---

## 1. Emotional Field Core
- 7D Mood → 2D 熱力学場 `E(x,t)` へ投影し、エントロピー/流束/拍動を常時計測。
- Ignition Index（前景化スコア）で「いま大事な刺激か」を判定し、応答の重みを自動で調整。

---

## 2. Multimodal Perception Bridge
- Consent-aware カメラ / マイク / テキストを `SenseEnvelope` へ正規化。匿名化・TTL・フィルタを標準装備。
- DeepSeek-OCR + Video Timeline で PDF/画像/動画を“章単位”に分解し、RAG へ即接続。

---

## 3. Memory & DreamLink
- MemoryPalace（L1/L2/L3）で感覚痕跡→エピソード→スキーマへ蒸留。Nightly ETL が日次で整理。
- DreamLink が情動地形から創発プロンプトを作成し、Diary や Catalyst へ送り直す。

---

## 4. Narrative・Community・Care
- StoryGraph + Community Orchestrator：ThreadGraph / Lore / Spoiler で会話の流れを維持。
- Inner‑Care Protocol：calm90 / boost60 / reset30 などのケアモード、HealthCard が体調を可視化。
- Chaos Taper：周期×振幅で“熱量の振れ”を抑制し、DET/LAM 指標で暴走を察知。

---

## 5. Sunyata Flow（縁起ログ）
- **DO‑Graph**：五蘊（色・受・想・行・識）の寄与を `causality/do_graph.py` で常時トレース。`receipt.sunyata.do_topk` に縁起 Top‑K を記録。
- **Self Posterior**：`mind/self_nonself.py` が役割確率と自己物語の整合度を推定。固定人格ではなく分布で“今の私”を説明。
- **Clinging Gate**：`config/sunyata.yaml` の閾値を超えたら Forgetting boost / URK 最小ステップ / Candor level を自動調整し、「手放す勇気」をコードで担保。
- **GO‑SC + Interference**：Nightly の Generalization Gate が価値の高い痕跡だけ TTL を延長。Interference Gate が似すぎるリプレイをマスクして暴走を防ぐ。

---

## 6. Referent Hint × Candor
- **Gaze referent_hint**：共同注視＋視線 LLE で「相手はいま◯◯を見ている」を抽出し、ハードドメイン外なら `【参照】…` をプロンプト冒頭へ。指さし無しの “これ／そこ／あれ” を即座に解決。
- **Candor Persona**：calm_candor / bold_resolver を disclosures.yaml で切替。Sunyata Gate から `candor_force_level` を受けたら warn/must/ask を最初に宣言し、“非共有の残り” を必ず伝える。
- **Kids Poster**：キッズ版ポスターにも「空（くう）の流れ」と「視線共有」を追記し、家族と一緒に学べる内容に更新。

---

## 7. Sensibility & Love Layer
- `Σ(t)` の拍動、ジャーク、RQA 決定性から“世界への合わせ方”を評価。
- `love_mode` が同期・自己開示・保護シグナルを合成し、Live2D / Robot / TTS まで一貫したふるまいを生成。

---

## 8. Formal Safety & Auto‑Tuning
- Litex Rewrite：`plugins/formal_reasoning/rewrite_rules/*.ltx` で pause / κ / warmth をルール宣言。
- Lean Invariants：`proofs/*.lean` と `enforce_invariants` が rate‑limit / containment を保証。
- 自動チューニング三段ギア  
  1. `scripts/autotune_replay.py` で Replay 方策を学習  
  2. `ops/canary_bandit.py` が小規模カナリア展開  
  3. Lean Gate で CI / runtime の逸脱を即修復

---

## 9. Nightly Optimization
- GO‑SC Gate：SWR 風スコア＋スキーマ利得で長期化する記憶を選抜し、`meta.ttl_scale` へ書き込み。
- URK 相転移：学習初期は逆再生（評価伝播）、熟達後は順再生（将来計画）へ自動シフト。
- QUBO-based Retriever Selection、Warm Cache、t‑digest / HLL で Warm/COLD 層を軽量化。

---

## 10. Evaluation・Dashboard
- eval/report_nightly.py：NDCG@10 / H4 / p95 / 再利用率 / Sunyata 指標を集計。
- ダッシュボード：Ignition・Sunyata・Gaze・Interference Gate を可視化。p95 レイテンシ、整定時間、誤点火を常時監視。

---

## 11. Quick Start
1. `ENABLE_COMMUNITY_ORCHESTRATOR=1` で StoryGraph / Diary を起動。
2. Perception Bridge + DreamLink を立ち上げ、Vision/Audio→`I(x,t)` を流し込む。
3. `formal_reasoning.enabled: true` に設定し、`pip install pylitex` / `leanproject get mathlib` を実行。
4. `python plugins/formal_reasoning/litex_demo.py` / `prove_controls.py` で制御ルールを検証。
5. `python scripts/autotune_replay.py --logs data/logs` で方策チューニング開始。

---

## 12. Roadmap（抜粋）
- Persona ごとの Rewrite DSL 拡張、lean lemma / love_mode bounds / Σ rate の公開。
- Bradley–Terry ベースの Replay weight 調整と Chaos Ensemble 可視化。
- Dashboard に Tuning Heatmap / Bandit Live / Lean Repair Histogram を追加。
- Sunyata 指標と実サービス KPI（NPS/苦情率）を結び付けた評価パイプラインを整備。

---

> EQNet は「感性・身体・意識っぽさ」をカオス×制御×記憶×倫理で運転する、“拍動する相棒”の工学版ロードカーです。
