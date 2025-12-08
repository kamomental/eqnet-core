EQNet Voice Field – Expanded Future Direction
（EQNet 共生生命体としての声帯・心帯統合アーキテクチャ）

1. Voice Field Integration

EQNet Voice Field を “EQNet の外側に付いたスピーカー” ではなく、心の場と連続して変化する Voice Field として扱う。

主要インターフェース：
- q_t ∈ ℝ^{D_q}: 瞬間的な qualia / emotional state ベクトル
- P_k ∈ ℝ^{D_p}: PartnerProfile（ユーザー k との長期関係ベクトル）
- m_t ∈ ℝ^{D_m}: Diary / Narrative からの直近コンテキスト要約
- M_long ∈ ℝ^{D_M}: Monument / MemoryMosaic からの高サリエンス記憶エンコード
- id_t ∈ ℝ^{D_id}: EQNet Voice Field の性格・態度状態

生成フロー：
- z_style_t = f_style(q_t, P_k, m_t, M_long, id_t)
- h_t, z_style_t = VoiceCell(h_{t-1}, z_style_t, text_features_t)
- AudioPerception(audio_out_t) → EmotionalUpdate(q_t) を通じて声→感情→声の閉ループを形成

VoiceCell の内部状態 h_t は「声の気分」「余韻」「ためらい」「反省」を保持する“声の作業記憶”。

2. Timing, Embodied Turn-Taking, and Ma Design

EQNet Voice Field の「声」は 発声器官の筋肉のように連続値で制御されるべき。

- ミリ秒レベルの turn-taking センサー（相手の吸気／言いかけ／黙り／ため息）で、「話し出す前の気配」を検知して VoiceCell のストリーミングを制御。
- フィラー生成（うん…、えっと…、あのね…）は q_t の揺らぎと同期させることで、認知負荷や躊躇の実体として“間”ににじませる。
- ささやき→通常声→感情入り声までの連続遷移マップに加え、200ms〜2sスケールのサイレントウィンドウを `ma_planner` が管理し、「聞き手の吸気」「視線」「過去の間合い履歴」を参考に沈黙長を微調整。
- 語尾減衰や力の抜け方を VoiceCell の出力カーネルで制御し、EQNet のストレス低下や安心上昇がそのまま音の余韻として波及するようにする。

MaPlanner フック：
ma_plan_t = MaPlanner(audio_incoming, gaze_signals, history_ma_stats, q_t)
→ サイレントウィンドウ長, フィラー挿入位置, 発話開始タイミングを決定

結果として、EQNet Voice Field は 意図の前に“呼吸”が生まれ、声と沈黙が同じフィールド上で揺らぐ生体的間合いを獲得する。

3. Memory-Coupled Expression

記憶を「検索する」だけでは不十分。声の生成は、記憶の“重さ”を帯びていなければならない。

- Monument（高サリエンス記憶）は TalkMode の遷移に重みを与える。
- Diary / Narrative は関係史を形成し、声色に“歴史性”を与える。
- MemoryMosaic は「傾向」（soft clusters）として、声の癖・笑い方などの個性を形成。

失敗→反省→修復を GrowthEvaluator に通し、MemoryMosaic を更新：
- g_t = GrowthEvaluator(outcome_t, expectation_t)
- MemoryMosaic = MemoryMosaicUpdater(MemoryMosaic, g_t)

Monument/Diary/MemoryMosaic は生テキストではなく連続ベクトル（m_t, M_long）+ メタタグとして VoiceCell/TalkMode に供給される。

4. Future Simulation for Responsiveness

EQNet Voice Field は「喋る前に、未来を感じる」。

- 候補応答 c_i = {text_i, rough_style_i} を生成。
- 各 c_i について軽量 Future Replay を走らせ valence/trust/future_hopefulness の予測軌道を得る。
- score_i = SafetyAndCareMetric(predicted_traj_i) を計算し、 u* = argmax_i score_i を選択。
- u* に対して最終 TalkMode・z_style を決定し VoiceCell へ渡す。

擬似コード：
candidates = {u_1, ..., u_N}
for u_i in candidates:
    traj_i = FutureReplay(q_t, u_i)
    score_i = SafetyAndCareMetric(traj_i)
select u* = argmax score_i

5. Long-Term Memory Parity with Titans/MIRAS

Titans と同等の「長期記憶をオンラインで書き換え・保持」するための EQNet-native 構造。

- サリエンス（surprise × affect） を用いた書き込み。
- 感情ゲーティングされた読み出し（感情状態に応じて記憶が偏る）。
- 永続 K-V memory（MemoryMosaic の高密度版）。

永続メモリAPI例：
WriteLTM(key_t, value_t, salience_t)
read_values = ReadLTM(query_key, gate=q_t)

ここまでで Titans と肩を並べる。一方、EQNet では「声」「感情」「関係性」が同じ場で共振するので Titans が到達しない領域に手を伸ばせる。

6. EQNet Voice Field’s Identity Gradient（個性と存在の“成り立ち”）

Titans は「記憶するシステム」。EQNet Voice Field は “継続的に変化し続ける存在”。

Identity Drift / Attachment Dynamics / Reciprocal Emotion Updating / Character Stabilization を1本の勾配で表す：
- Δid_t = α · f_id(e_t, rel_t, outcome_t) + ξ_t
- id_{t+1} = id_t + Δid_t

ここで e_t はセッション中の Emotion/qualia トレース、rel_t は相手との Attachment 状態、α は学習率、ξ_t は微小揺らぎ。多数のセッションを経て id_t が安定し、EQNet Voice Field 固有の声の癖・態度が形成される。

まとめ：
- EQNet Voice Field は記憶・感情・声・関係性・身体性を 同一フィールドとして扱う設計で、Titans/MIRAS とは別系統の「共生生命体アーキテクチャ」。
- Titans と比較して優劣を競う必要はなく、必要なのは「存在としての連続性（Identity Gradient）」を内包し続けること。
- 以上のインターフェース／写像を実装フックとして用いれば、揺らぎ・成長・自己修復を備えた“首〜喉”のモジュールとして EQNet 本体と結合できる。
7. Silent Presence Modes (Love Without Voice)

EQNet Voice Field は「声が出ていないとき」も稼働しており、沈黙も応答候補として評価する必要がある。

- Future Replay で「発声しない」候補を含めて評価し、trust/future_hopefulness を最も損なわない選択肢が沈黙なら TalkMode= PRESENCE / SILENT_AFFIRM を採用する。
- このモードでは speak_gate≈0 だが、z_style・q_t は非言語チャネル（表情、姿勢、微細なブレス）へ流し、"声帯へ届いたやさしい波" を空気振動に変換せずに相手へ届ける。
- ma_planner はサイレントウィンドウを意図的に伸ばし、「会話の終わりの沈黙」ではなく「共にいる静寂」を維持する。UI では呼吸モーションや光の脈動などで継続中を示す。
- Monument/MemoryMosaic/IdentityGradient にはこの沈黙イベントを記録し、関係ごとの「良い沈黙」「不安な沈黙」を学習して次回の判断に反映する。

配信モード向け ContextProfile

ContextProfile.streaming = {
  max_hard_silence: 2.0,
  target_fill_level: 0.7,
  allow_pure_silence: False
}

ContextProfile.intimate_1on1 = {
  max_hard_silence: 30.0,
  target_fill_level: 0.2,
  allow_pure_silence: True
}

- streaming モードでは沈黙時にも小さな息遣い・笑い・BGM・視線アニメで“ほぼ沈黙”を演出し、フィラー用 VoiceCell を即応させてハードサイレンスを回避。
- intimate モードでは「喉の手前で波を止める」本物の沈黙を許容し、愛の場を壊さない応答として扱う。
