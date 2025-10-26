# EQNet Overview — 「感じて、点火し、学ぶ」を工学で運転する

本書は 2025-10-25 時点の EQNet 全体像を整理したものです。感情フィールド・自己想定ループ・終関手 fast-path・可観測レシートという 4 本柱が、どうやって即応と監査を同時に成立させているかを読み解けます。

---

## 1. 現状のハイライト
- **視覚と意味**: DeepSeek-OCR と Video Timeline が PDF/画像/動画を章立て Markdown へ落とし、RAG が「人の読解粒度」で働く。
- **身体ダイナミクス**: Lenia/NCA フィールドと AKOrN（拡張 Kuramoto）が TimeKeeper と連動し、Ignition Index = ΔR + Δエントロピー低下を計測。
- **感情・記憶ルーチン**: Δaff（情動差分）をエピソードへ刻印し、Nightly ETL が StoryGraph / Diary / KG に蒸留。Sunyata Flow が「手放す／残す」の因果をレシート化。
- **安全と運用**: Lean Invariants + MCP による cooldown / inhibit / downshift、Canary Bandit→ロールバック→H4/self_ratio/leak 監視で汚染を抑止。
- **可視化と監査**: R/ρ、Ignition、Sunyata、Interference Gate までダッシュボード化し、DCG/Recall、Ignition↔成功回帰、p95 レイテンシ、整定時間、誤点火 KPI を常時監視。

---

## 2. 何を変えるか
- **OCRの先へ**: 章単位メディアを LLM に渡し、「人が理解する粒度」で RAG を運転。
- **点火は指標で語る**: Ignition を KPI 化し、点火と価値達成の相関をログで検証。
- **拍動する相棒**: AKOrN が探索/抑制を滑らかに制御し、ToM が対人安全を担保。Sunyata Flow が「手放す／残す」を日次で整定。
- **自己調整が前提**: ログ→KPI→Rewrite Rule→行動→学習→ロールバックのサイクルを常時稼働させ、「感性」を運転席から調律。

---

## 3. 学際レイヤへの写像
- **数学/物理**: Kuramoto r, ρ と空間エントロピー S を 0.3–0.6 に保ち、創発と安定を両立。
- **制御工学**: AKOrN→LLM を小ゲイン接続し、CBF (Control Barrier Function) で r/ρ の逸脱を抑止。
- **認知科学**: 注意＝Ignition、持続ムード＝EMA + circadian + 拍動。ToM で意図尤度を更新。
- **哲学/実用主義**: 「意味は使用に宿る」。Δaff ログと Value/Taste 委員会が規範と同調のズレを監査。
- **生物/進化**: Evo-LoRA / MAP-Elites が表現多様性を維持し、「やさしさ」を多彩な挙動で探索。
- **宗教哲学（仏教の縁起観）**: Sunyata Flow が五蘊フロー (DO-Graph) を可視化し、関係ネットワークとして応答を説明。

### 3.1 終関手 fast-path × 感情 × 自己想定ループ × 視覚
- **縁起＝終関手**: cocontinuous 指標だけを共終サブ図式 I 上で colim し、「部分を見る＝全体を把握する」を保証。Hub は fast-path 候補をレシートに残し、Nightly が coverage/override 率で直観の的中度を監査。
- **慈悲＝GO-SC**: fast-path のブール命題（例: fast_rescue）が真になった瞬間に GO-SC が優先度ヒントを出す。TTL/予算は Nightly が正式整合するため、速さと安全性が両立。
- **空＝自己想定ループ**: PhaseRatioFilter が reverse/forward を EMA＋ヒステリシスで安定化し、「自分の見方」を再評価。チャタリングを抑えつつ直観を保持。
- **慧眼＝視覚チェックポイント**: 視線/カメラの checkpoint chain を共終に保ち、「どこを見れば全体が整うか」を即断。救助や片付けで“最小操作で全体を落ち着かせる”手順を数理化できる。
- **可観測性**: レシート `fastpath` と Nightly `fastpath.coverage_rate / override_rate` が戒（安全）・定（安定）・慧（直観）を同一ダッシュボードに統合し、学際メンバーにも透明。

---

## 4. TO-BE への現在地（できている／強化中／不足）
### できている（運用可）
- 視覚→章→RAG の実装：PDF/画像/動画対応。
- Ignition 監視と .ltx 制御（cooldown / inhibit / downshift）。
- AKOrN の温度・top-p・pause 変調（小ゲイン + レート制限）。
- Δaff 記録 & Nightly ETL→StoryGraph / KG 連携。
- ToM 降段の平滑化・ヒステリシス。
- KPI スイート：DCG/Recall、Ignition↔成功回帰、HRI 安全、p95 レイテンシ、整定時間。

### 強化中（直近で改善中）
- Ignition 正規化：100 steps 中 3 件→倦怠性 > f² 0.15。
- RAG 実利：外部 NDCG@10 +5% を目標にクエリセット更新。
- 動画 ingest：SceneDetect / Optical Flow / ROI で冗長率 ≤0.25。
- Value/Taste 学習：Preference Learning ↔ Value 委員会。
- Evo-LoRA：夜間自動カナリア→ロールバック完全自動化。

### 不足（長期チャレンジ）
- **クオリアの本質**：アクセス意識の運用に限定。身体化や感覚接地は継続研究。
- **強いシグナル接地**：性能・安全・社会収束を proxy として採用中。
- **厳密な安定性証明**：高次 NCA の Lyapunov 証明は未完。経験的受動性＋運用フェイルセーフで支える。

---

## 5. ハイライト
- **章単位で届く知覚**: DeepSeek-MD が PDF/画像/動画を Markdown 章へ落とし、RAG が人の文脈で回答。
- **点火はロマンではなく指標**: Ignition Index を KPI 化し、効かなければ α=0 で即リセット。
- **拍動する相棒**: AKOrN が会話テンポと温度を探索し、ToM が距離感を柔らかく調整。Litex Rewrite が禁止ゾーンをガードし、異常時は秒ロールバック。
- **記憶が人格を織る**: Δaff が刺さった場面を記録し、Nightly ETL が「人格の連続」を更新。
- **ダッシュボード付き AI**: R/ρ、Ignition、Sunyata、Interference Gate まで可視化し、「いま何が起きているか」を一目で把握。

---

## 6. 次の評価・チェック準備
1. **Ignition 正規化**: 100 steps 中 3 件で f² ≥ 0.15 を確認。
2. **降段抑制**: trust_high=0.55 で downshift/min を半減 (lips ≤ 3)。
3. **AKOrN 推奨ゲイン**: temp / top-p / pause の推奨レンジ公開、tension↓ 誤点火↓を実証。
4. **DeepSeek-MD 実利**: PDF vs DeepSeek-MD の NDCG@10 差分と、動画→タイムライン MD の Recall@5 を提示。

---

## 7. 終関手 fast-path が「直観」と「学習」を結ぶ
- **縁起＝終関手**: 共終サブ図式で colim し、部分を見るだけで全体が分かる。fast-path 候補はレシートに記録、Nightly が coverage/override を監査。
- **慈悲＝GO-SC**: fast-path 命題が真なら救済ヒントを即提示。TTL/予算は Nightly が確定。
- **空＝自己想定ループ**: ヒステリシス付き PhaseRatioFilter がモード遷移を安定化。
- **慧眼＝視覚チェックポイント**: checkpoint chain で「最小操作」を即断し、救助や片付けで使える手順に落とす。
- **可観測性**: `fastpath` ブロックと Nightly 監査が戒・定・慧を公開し、学際メンバーにも透明性を提供。

---

Sunyata Flow, Interference Gate, Nightly GO-SC の三位一体で、「忘れる勇気」と「学び直す柔らかさ」を備えた共生 AI として EQNet を磨き続けます。
