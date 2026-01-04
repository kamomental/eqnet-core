# EQNet 進捗スナップショット（README / Roadmap基準）

目的
- README / roadmap の記述に対して、現時点の到達度を短く示す
- 実装済み / 進行中 / 未着手 を一目で把握できる状態にする

進捗サマリ（結論）
- **運用可能**: de viant / resonance の分離運用が成立し、Nightly→提案が実データで閉じた
- **進行中**: 感覚入力の拡張・Ignition正規化・Value/Taste学習
- **未着手**: Qualiaの形式安定証明や長期倫理対話の体系化

README の Current Status に対応
Operational（運用段階）
- RAG統合（PDF/画像/動画）: 実装済み
- AKOrN制御 / Ignition監視: 実装済み
- Δaff ログ / KPI suite: 実装済み
- ToM安定化: 実装済み
- Nightly監査: 実装済み

Improving（改善中）
- Ignition正規化: 進行中
- Value / Taste学習: 進行中
- DeepSeek-MD メディア取り込み: 進行中
- Evo-LoRA rollback: 進行中

Ongoing Challenges（未解決）
- Qualia表現: 未解決
- 感覚グラウンディング: 未解決
- NCAの形式安定証明: 未解決
- 倫理対話 / 社会的アンカリング: 未解決

Roadmap（3波構成）に対する進捗
Wave 1（即時応答・観測の安定化）
- 短期の観測・監査ループ: 実装済み
- Nightly / KPI / 可視化の定着: 実装済み
- deviant / resonance 分離運用: 実装済み

Wave 2（共鳴・価値学習の安定化）
- Value / Tasteの学習系: 進行中
- 文化的補正（culture.yaml）: 実装済み
- 連続運用の安全ガード: 実装済み

Wave 3（長期進化・説明性の強化）
- 長期記憶の統合: 進行中
- 進化的反省（Evo / Weekly）: 進行中
- 形式的な安定証明: 未着手

EQNet-core 検証の到達点
- boundary と decision の分離が数値で確定
- deviant / resonance の分離運用が成立
- Nightly → 3日窓 → 提案が実データで閉じた
- 運用値（w_reward / w_risk / beta_veto 等）が暫定固定

次に整理すべき未解決タスク
- decision 入力の正規化（risk/uncert のスケール校正）
- boundary を意思決定へ昇格させるかの設計判断
- resonance を安全KPIとして定常運用するフローの固定

# EQNet Progress Snapshot (2025-12-27)

現在地（確定）
- eqnet-core: boundary と decision の分離が数値で確定し、deviant / resonance の分離運用が成立
- Nightly → 3日窓 → 提案の監査ループが実データで閉じた
- LazyGraphRAG: Graph → ID → JSONL → context の retrieval が成立し、短い運用版の回帰テストで PASS を確認済み
  - PowerShell 実行環境では正規表現の日本語レンジが崩れる場合があるため、Unicode エスケープ版で運用する

進行中
- decision 入力の正規化（risk / uncert のスケール校正）
- Value / Taste 学習の強化
- 長期記憶の統合と運用ルールの固定

未着手 / 要判断
- boundary を意思決定へ昇格させるかの設計判断
- Qualia/倫理対話の体系化（長期説明性）
- 感覚グラウンディングの形式安定（証明/主張の置き方）

次の一手（優先度順）
1) boundary は意思決定に昇格させない（入力ゲートとして最上位）
- boundary = Do/Don't を決める強制ゲート（HOLD / 縮退 / 停止 / 要確認）
- decision = “やるならどうするか” を決める最適化（優先順位・手段・配分）
- 例外は TTL 付き・監査ログ必須・再評価必須（Nightly + 3日窓）でのみ許可

2) risk / uncert を分離して正規化（0〜1）し、ゲートと意思決定を分ける
- uncert: 情報不足/曖昧さ/観測の弱さ（探索・質問・追加観測で下がる量）
- risk: 失敗時の損失（安全・信頼・コスト、可逆性・曝露で決まる量）
- gate 例:
  - risk 高 & uncert 高 -> HOLD（保留/追加情報要求）
  - risk 低 & uncert 高 -> EXPLORE（質問/プローブ/シミュレーション）
  - risk 高 & uncert 低 -> HUMAN_CONFIRM（高確信だが危険）
  - risk 低 & uncert 低 -> EXECUTE（実行）

3) 監査ループで再校正（Nightly → 3日窓）
- 観測: decision 結果（成功/失敗/保留の妥当性）と deviant/resonance 分解
- 更新: 閾値・重み・例外TTL の微調整（比較可能性を保ちつつ小さく更新）
- 監査: “境界ゲート” は不変、decision 側のみ校正対象とする

補足（運用）
- 長期記憶は混ぜない
  - Graph = 関係性
  - JSONL = 一次ログ
  - RAG context = 説明材料（再現可能な引用）
