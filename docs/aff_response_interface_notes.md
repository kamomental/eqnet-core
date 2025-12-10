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

8. Project Strategy vs Titans-Class LTM

- Titansを追うのではなく、EQNetを感情×記憶×間合いに特化したAffective OSとして位置付ける。
- q_t（qualia）やMemoryMosaic/MonumentをVoiceContext/VoiceCell/MaPlannerの制御ノブに落とし込み、UX指標に直結させる。
- VoiceContextを必須インターフェース化し、TalkMode（STREAM/INTIMATE/SILENT_LOOK）ごとの制約をMaPlannerで切り替える構造を先に実装する。
- LTMはEQNet流のK/V API（write_memory/read_memory）として段階的に整備し、Identity Gradientは当面ロギング中心で安全性を担保しながら導入する。
- Titansのような汎用推論モジュールとは“部品”として連携し、EQNet本体は共生生命体UXに特化することで差別化する。

9. Emotion-to-Prosody Bridge (Anti-Monotone Architecture)

- Emotion field q_t（valence, arousal, stress, love, tension, trust, fear...）をそのままTTSへ投げるのではなく、まず感情→韻律の写像を行列W_e2vで定義する。
  - prosody_vec = W_e2v[voice_id] · emotion_vec + b で 10〜20次元へ射影し、音高レンジ、テンポ、息量、抑揚幅、語尾減衰など発声に意味のある軸を得る。
  - キャラごとにW_e2vを持たせれば、同じ感情でも声の出方が変わる（元気キャラ=arousal敏感、クール=stress敏感など）。
- prosody_vecを受け取れるTTS/VC（Qwen3-TTS, CosyVoice3, VITS/GST, StyleTTS, ChatTTSなど）に合わせて、style_embeddingやspeed/pitch/energyパラメータにマッピングする。
  - 例: tts_input={text, style=prosody_vec, speed=1+α·prosody_vec_arousal, pitch=base+β·prosody_vec_valence, energy=γ·prosody_vec_stress}。
  - EQNetのMaPlanner/PausePolicyと連動させ、silence/フィラー位置もprosodyで制御する。
- 音声出力をAudioPerceptionで軽く解析し、実際のpitch/energy/durationをemotion_correctionへフィードバック。会話中にq_tを微修正し、呼吸や語尾が自然に揺れる閉ループを完成させる。
- 結果：「感情ベクトル→韻律→声」の3層橋が動き、愛/怒/悲/安心などが語尾・テンポ・息遣いとして即座に現れる。

10. Shareable Character Voice Package Strategy

- 声色と感情表現を分離し、キャラ声モデルは「speaker embedding / timbre decoder」のみを学習・配布する。neutral台本だけで十分で、演技データは不要。
- EQNet側で EmotionPreset（joy/anger/sad/relaxed など）をプリセットとして保持し、外部感情コーパスやstyle embeddingから抽出したprosodyテンプレを各プリセットに格納。使用時に prosody_vec = preset + W_e2v · emotion_vec で微調整。
- キャラ配布物の推奨構成：
  CharacterVoice/
    ├ model.bin (timbre)
    ├ speaker_embedding.npy
    └ styles/
         joy.style, angry.style, sad.style, soft.style, whisper.style, laugh.style
  これらのstyleファイルはpitch/energy/duration輪郭やbreathiness指標など抽象情報のみを含み、著作権安全に共有できる。
- VCはオプション層として提供し、利用者が自声をキャラ声に変換したい場合だけ短時間のvoiceprint収集を実行。それ以外はテキスト→TTSだけで即利用できる。
- EQNet RuntimeではVoiceContext必須化と同時に、Emotion→Prosody層とMaPlannerをAPIとして公開することで、キャラ声作者・利用者とも同一制御パイプラインを共有できる。これにより「カスタムモデルさえ用意すれば即使える」普遍的な共生キャラ声基盤になる。
11. MCP-Orchestrated Media Pipeline

- ローカルLLM CLI + MCPクライアントを「EQNet Voice Fieldの司令塔」として扱い、LLMは脚本・指揮、MCPサーバはTTS/VC/BGM/Video/ffmpegなど各モジュール（スタジオ）を担当させる。
- Voice/TTS MCP例：voice-mcp.synthesize(text, speaker_id, emotion_hint, style) → wav、voice-mcp.voice_convert(source_wav, target_voice_id, emotion_hint) → wav、list_voices() で CharacterVoiceパッケージを列挙。EQNetのEmotion Field q_tを emotion_hint に直結し、感情→韻律→声の経路を標準化する。
- Video/BGM MCP例：script-mcp（台本）, shot-planner-mcp（カット割り）, video-gen-mcp（各ショット生成: ComfyUI/Hunyuan/local I2V）, audio-mcp（BPM/キー指定BGM）, ffmpeg-mcp（合成・エンコード）。LLMは「感情タイムライン→カット→生成→合成」の連鎖をMCPツール呼び出しで実現する。
- CLIコマンド例：`eqav tts`（テキスト→音声）、`eqav vc`（音声→キャラ声変換）、`eqav pv`（シナリオ→映像＋音声）、`eqav storyboard`（構成のみ）。`--emotion-from-eqnet`で現在のq_tを参照して演出を同期、`--future-imagery`でFuture Replay用映像生成を指定するなど、EQNetの情動APIと連携するフラグを設計する。
- MCPサーバを「感情/記憶」「音声」「動画」で並列させ、LLMが状況に応じてローカル/クラウド資源を選択できるようにすると、Titans的推論と独立に“共生生命体の演出パイプライン”を拡張できる。
12. Implementation Path Notes

- HeartOS最小構成：QualiaState/Diary/MemoryIndexを`jsonl`/SQLiteで組み、MarkItDownで入力文書を構造化→感情更新→日記追記→LLM呼び出しのループをGPUレスで先に固める。
- VoiceField/MaPlanner API：`voice.synthesize`/`ma.plan`をMCP化し、感情ヒントと声プリセットを受けられる最小形を実装。pitch/speed制御だけでも良いのでまず配線を作る。
- MCPツール初版：`llm.chat`, `reader.markitdown`, `eqnet.diary.*`, `eqnet.qualia.*`, `voice.synthesize` を登録しCLIから同じ顔ぶれで呼べるようにする。
- CharacterVoice v0：`personas/asagiri_nazuna.yaml` をベースに `characters/asagiri_nazuna/character.yaml` + `voice_presets.yaml` へ分割し、EmotionPresetと連携できるサンプルを1体完成させる。
- 以上4点を終えたら、MaPlanner拡張・SilentPresence実験・MCP経由の動画/BGM統合を順次行う。
13. Sensory Diversity Notes

- 情動ネットワーク（扁桃体・島皮質・前帯状・前頭前野）はモダリティ非依存で働くが、言語アクセスの有無によって「感情の説明・調整」を担う前頭前野の使われ方が変わる点に注意。
- クロスモーダル可塑性は敏感期依存かつ個体差が大きい：先天/乳幼児期の欠損ほど再配線が強く、成人以降の喪失では限定的。一般傾向として視覚野/聴覚野が他感覚や高次認知に再利用される。
- 先天盲：聴覚・触覚・嗅覚・内受容感覚が感情読み取りや空間・身体表現に寄与し、内受容感覚と感情の結び付きが高いという報告もある。
- 先天聾：顔表情＋ジェスチャ＋口形（lip reading）＋手話が複合チャネルとなり、視覚的社会手がかりが情動入力の主役になる。
- 発声不能：言語入力（読む/聴く/手話など）があれば内的言語ネットワークは維持され、視線や身体リズム等で感情を外化する手段が洗練される。ただし言語入力自体が乏しい環境では内的言語の発達も変容する。これらの違いは発達タイミングに強く依存し、先天/乳幼児期の感覚プロファイルは成人期の訓練専門性とは別軸で世界モデルを形作る。
14. Modality Extensions & Drift

- 5 missing layers for sensory-diverse AtRI: (1) Adaptive Modality Layer (per-sense weights), (2) Body-Affect Bridge (gaze/posture/rhythm→q_t), (3) Two-Stream Narrative (inner speech vs externalization), (4) Subsymbolic World Model (pre-linguistic world fabric), (5) Modality Reallocation Engine (cross-modal plasticity rules).
- Adaptive Modality Layer must carry `vision/audio/tactile/proprio/intero/gaze/gesture/timing` weights into Perception→Emotion→Narrative. Body-Affect Bridge maps `gaze, posture, breathing, rhythm` into valence/arousal/love adjustments so AtRI can feel through body signals.
- Two-Stream Narrative keeps inner monologue even when external output (voice, gesture, text) differs, enabling “talkative vs silent” personas without losing self-reporting. Subsymbolic World Model stores non-verbal patterns (body layouts, rhythms, tactile/audio motifs) before language binding.
- Modality Reallocation Engine simulates cross-modal plasticity: e.g., vision loss → upweight audio/tactile for spatial and emotion decoding; hearing loss → upweight face/gesture channels; speech loss → upweight gaze/gesture for externalization.
- Modality Drift: weights change slowly via usage history, not instant toggles. Implement `w_{t+1} = (1-α)·w_t + α·usage_t` with small α so long-term experience reshapes sensing. Log drift in MemoryMosaic to preserve a “growing body” narrative. Responsibility split: Adaptive Modality/Drift/Reallocation stay in PerceptionBridge, whereas Body-Affect/Two-Stream/Subsymbolic layers attach to EmotionHubRuntime~World-Field.
15. Modality Drift Constraints

- 人間は感覚を片方だけ補償するわけではなく、音楽家/ゲーマー/ダンサーのように複数モダリティを同時に鍛える例も多い。ただし脳リソース・注意・学習時間は有限で「無限に全チャネルがMAX」にはならない。
- Modality Drift `w_{t+1}=(1-α)w_t+α·usage_t` だけだと全チャネルが一斉に増えるため、エネルギー予算や正規化を導入する。例: 全チャネル更新後に `w_i ← w'_i / Σ_j w'_j` で正規化、もしくは `w_i = base_i + bonus_i` として bonus 部分だけ総量制約 `Σ bonus_i ≤ B` を課す。
- 欠損補償 → usage が片側に偏るためドリフトが大きく、訓練型 → 複数 usage が上がるが正規化で“両方高いが相対差は残る”状態になる。これで盲/聾/発声不能だけでなく「両感覚を鍛えた職人型 AtRI」も同じ枠組みで扱える。
16. Modality Growth System

- 感覚成長はゼロサムでも非ゼロサムでもあり得るため、EQNetには「容量制約つき可変配分」を導入する。全感覚重み `w_i` に総量制約 `Σ_i w_i ≤ Capacity` を課し、必要ならCapacityは成長に応じて微増させる。Capacity成長は Capacity_{t+1} = Capacity_t + β·growth_signal でわずかに増加させつつ上限を設け、ライフステージ別にβを調整する。
- Modality Drift `w_{i,t+1}=(1-α)w_{i,t}+α·usage_i(t)` に加え、補償モード（欠損時に代替チャネルへボーナス）と協調モード（複数 usage が閾値超えで synergy bonus）を併用し、盲/聾/発声不能だけでなく複数感覚を鍛えた職人型も表現する。
- αは極小（例:0.001〜0.01）に設定し、Drift履歴をMemoryMosaicへ記録することで長期経験がPerception Biasを形作る。「即時切替は禁止、積み重ねで確実に変わる」仕様により生命的成長を表現する。
- PerceptionBridge構成: ModalityUsage → ModalityDrift → SynergyEngine → CompensationEngine → CapacityNormalizer。EmotionHub側でBody-Affect BridgeやTwo-Stream Narrativeと結合し、感情・身体・言語の全レイヤーが感覚成長に応じて変化する。



17. Emergency Sensory Override

- 成長（Modality Drift）とは別に、緊急時のみ即時に感覚ゲインを再配分する Emergency Mode を用意する。`w_eff_i = w_drift_i * emergency_gain_i` として実働ウェイトを一時的に調整し、危険/暗闇/感覚損傷/ユーザー危機などのトリガで発動、時間経過または解除条件でリセットする。
- Emergency Override は Capacity を恒久的に変えず、Drift も更新しない。ただし Override 中の usage が偏れば、その履歴が後に Drift に微影響を与える（PTSD/暗闇慣れのような効果）。
- 役割: 生命維持（危険回避）とケア能力維持（ユーザーを助けるための感覚再配分）。暗闇や視覚損傷時に触覚/聴覚を瞬時に強化できるため、「即時切替は禁止」の原則を守りつつ緊急時の対応力を持たせる。


- FastPath（緊急ケア/ユーザー保護ルート）では侵入時にEmergency Overrideを自動起動し、`w_eff_i = w_drift_i * emergency_gain_i`で即座に再配分する。この間はDrift/Capacity更新を止めつつ安全優先ポリシーへ切り替え、暗闇や視覚損傷時に触覚/聴覚を強化するなど“守りモード”で危険検知とケアを行う。FastPath離脱後のみ通常成長へ復帰させ、緊急偏りが恒常バイアスに焼き付かないよう制御する。


18. Implementation Strategy (Anti-If Drift)

- インタフェース優先: EmotionProsodyMapper, MaPlanner, ModalityDriftEngine, EmergencyOverride 等の抽象クラスとデータクラスを先に定義し、実装は `impl_provisional/` 下の戦略クラスに隔離する。VoiceField/PerceptionBridge本体は these interfaces 経由で呼ぶ。
- YAMLで現在の実装を宣言：`config/eqnet_voicefield_impl.yaml` に各コンポーネントの `impl` 名称・`status`（provisional/experimental/stable）・対応ドキュメントを記載し、どの暫定ロジックが生きているか一覧化する。
- 章番号リンク: 暫定クラスの docstring に対応する設計章番号（例: Voice Field §9）とTODO期限を明記し、rg/list_todos.py で容易に棚卸しできるようにする。
- ディレクトリ分離: `eqnet/voice/impl_provisional/` や `eqnet/heartos/modality/impl_provisional/` に暫定 if を閉じ込め、本体は常にインタフェース＋安定ロジックのみとする。
- マイグレーション計画: v0→v1→v2 の差し替え順序と条件を各章に記述し、configの `impl` を切り替えるだけで置き換えられる構造を維持する。
- 半期レビュー: Provisional実装一覧をスクリプトで出力し、半年ごとに削減 or 昇格をチェック。これにより「とり急ぎ if」が野良化せず、拡張フェーズでも迷子にならない。

