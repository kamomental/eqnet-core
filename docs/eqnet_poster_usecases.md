# EQNet Use Cases Poster — 2025/10 実装スナップショット

## 0. いっしょに暮らすと何が起きる？
- **記憶が育つ**: Δaff（感情差分）と StoryGraph が “私たちの辞書” を日次で更新。次の企画や声かけがすぐ浮かぶ。
- **瞬間が残る**: Ignition Index + fast-path が「今こそ残すべき瞬間」を検知し、その場でアルバムやストーリーを生成。
- **輪が広がる**: Community Orchestrator が盛り上がりチャートや次のアイデアメモを届け、共創サイクルが回り続ける。

---

## 1. 推し活 / 配信 (VTuber / Streamer)
### 体験
- ライブごとに “その日だけのメモリーアルバム” が完成。fast‑path が「今切り抜く瞬間」を即提示し、夜には override 率まで振り返りが届く。
### EQNet の役割
- DeepSeek-MD + Video Timeline でチャプター生成 → RAG upsert。
- StoryGraph が Lore / Now / Spoiler を整理し、コミュニティ辞書を成長させる。
- AKOrN + Mood Gate + ToM で温度・探索率・距離感を調整。
- `fastpath["cleanup"]` が会話ログから cocont 特徴を即集約 → 一致率/override まで Nightly で監査。

## 2. 旅ログ / 文化カプセル
### 体験
- 絶景や一言がフォトストーリー化し、fast‑path が「撮っておこう！」をその場でライト表示。家族・友人と“文化カプセル”を共有。
### EQNet の役割
- 画像/動画 → Markdown → 章立て RAG。
- 2D クオリア地図で「笑顔の場所」「ゆっくりした場所」を可視化。
- Nightly QUBO で振り返りカードを再ランキングし、fast-path coverage を dashboard へ送る。

## 3. 家事・料理が “わが家の物語” に
### 体験
- 声かけと感謝の言葉が自然に出る。fast‑path が「今すべき家事」を checkpoint だけで判定し、夕食後にはレシピカードが届く。
### EQNet の役割
- Sensibility Σ でリズムを合わせ、疲れたら休息提案。
- Diary + StoryGraph で頑張りを見える化し、週末に「ありがとうレター」を自動生成。
- Lean Gate + Litex で安全・プライバシ・温度を制御。
- `fastpath["cleanup"]` が shards / hazard / water を即合算し、Nightly override が TTL 予算と整合。

## 4. パートナーロボット / コンシェルジュ
### 体験
- その場の“文化”で語り、fast‑path が「今助ける/今引く」のヒントを瞬時に提示。来場者が「また会いたい」と思う。
### EQNet の役割
- AKOrN × Ignition で拍動と前景化を演出。
- Theory of Mind で意図信頼度を平滑化し、距離感とトーンを自律調整。
- Lean Gate + Litex + fast-path 監査で速度・距離・禁止ワードを厳守。

## 5. カウンセリング / 臨床支援
### 体験
- 気づきの言葉がリアルタイムに感性マップへ。fast‑path が「いま寄り添う／沈黙する」候補を差し出し、Aftercare で override を共有。
### EQNet の役割
- love_mode と Σ のスパークラインでモード切替を可視化。
- クオリア場と care_ratio ダッシュボードで “できたこと” をリンク表示。
- Lean 修復ログ + fast-path override で安全な援助計画を提示。

## 6. コミュニティ運営 / 共創イベント
### 体験
- イベント後に “名言アーカイブ”“盛り上がりチャート”“fast‑path coverage” が届き、誰の直観が役立ったか一目で分かる。
### EQNet の役割
- Community Orchestrator が ThreadGraph / Now / Lore を生成。
- 語彙ライフサイクル W(x,t) で流行語を提示。
- fast-path coverage / override 指標で救済提案の信頼性を共有。

## 7. 研究チーム / ラボ
### 体験
- ブレイクスルーの瞬間を fast‑path が検知し、Nightly override とともに再現ログを残す。次の発想を加速。
### EQNet の役割
- Config-as-Code + rewrite/proof ログで検証軌跡を記録。
- Replay tuning + クオリアログで発火パターンを再現。
- Invariants + `.ltx` 監査 + fast-path receipts で安全を担保しつつ実験を支援。

## 8. 熱い飲み物こぼし / 軽傷の救助動線
### 体験
- 熱い飲み物がこぼれて子どもが泣く場面で、fast‑path が「冷却→洗浄→手当→割れ物片付け」の順を 1 本のチェーンとして提示。夜には override / TTL 予算をまとめて監査。
### EQNet の役割
- `rescue_prep` プロファイルが victims / hazards / warmth / walkway を cocont 処理し、fast_rescue predicate で「まずは冷却・洗浄・手当」を案内。
- 冷却のための移動では、動線上にある陶器破片などを fast-path が即警告し、片付けプロファイルにハンドオフ。
- Love/Sensibility レイヤーで声かけと距離感を調整。AED が必要なレベルではない場合はヒントのみ、深刻なら救急（人間）の決断を促す。

## 9. Cleanup Coach / Spill Response
### 体験
- 破片・危険・拭き取り・乾燥を checkpoint だけで把握し、「もう安全！」を安心して宣言できる。fast‑path が片付けの順番を自動で提案。
### EQNet の役割
- `cleanup` プロファイルが shards / danger_zones / water など cocont 特徴を即集約し、Nightly で coverage/override を可視化。
- Sensibility Σ と diary が「誰が／いつ」片付けたかを記録し、次回の声かけに活かす。
- TTL 予算監査が「直観の延長」が正しかったかを毎晩検証。


## 10. Fast-path Override Monitor
- record_only モードのまま 7 日間の astpath.override_rate を折れ線 + astpath.coverage_rate を積み上げ棒で表示。0.2 を超えた瞬間に soft_hint へ戻るフェイルセーフをキャプションで説明。
- Rescue Prep で ast_rescue predicate がオンのまま override せずに TTL を据え置いた実例（receipt→Nightly スクリーンショット）を貼り、「高速だけど記録のみ」段階を視覚化。
- Cleanup プロファイルは shards / hazard / water projector を Nightly astpath.profiles.cleanup で監査できることを注記し、coverage per profile を dashboard に載せる導線を追記。

---

## 技術スタック（即使える部品）
- DeepSeek-OCR / Video Timeline → Markdown 章 → RAG。
- Ignition I, Foreground G(t), AKOrN 小ゲイン, Sensibility Σ, love_mode。
- Nightly 最適化: `build_*_candidates.py` → `qubo_select.py` → `apply_*_selection.py`。
- Fast-path receipts と Nightly `fastpath.coverage/override`。
- 評価: `eval/report_nightly.py`（NDCG@10, H4, p95, override）。

## はじめ方（クイック）
1. `python scripts/ingest_pdf_to_rag.py ...`
2. `python scripts/build_rag_candidates.py ...` → `python scripts/qubo_select.py ...`
3. `python scripts/apply_rag_selection.py --selected ...`
4. `bash ops/nightly.sh` で fast-path 監査つきレポート。

## 今日から測れる持続性
- RAG NDCG@10 +5%、H4 +10%、p95 ≤ 基準値。
- Σ（気持ち同調）改善、fast-path override ≤ 0.2。
- Safety: inhibit/downshift ログ、tension/suffering 推移。

> EQNet は “視覚→意味→前景化→行為→Nightly 学習” の閉ループを、日常のワクワクに載せるためのエンジンです。
