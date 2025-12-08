### StreamingVLM-style Sensor Pipeline (StreamingSensorState)

EQNet-Hub では、StreamingVLM が提案する「注意の井戸（sink）＋短視覚ウィンドウ＋長テキストウィンドウ」という構造をセンサー処理に導入する。現状の gradio_demo_* では、webcam / 音声 / mediapipe から連続ストリームとして

- has_face
- voice_level / is_speaking
- pose_vec（骨格ベクトル）
- emotion_vec（表情・情動スコア）

などを取得しているが、これらを瞬間値のまま扱うとノイズに引きずられやすい。そこで Hub 側に `StreamingSensorState` を導入し、

- **sink_vec**: 非常にゆっくり変化する「環境・その人らしさ」の基準ベクトル
- **short_window**: 直近の複数フレーム（例: 32 フレーム）のセンサー列
- 上記を統合した **fused_vec**: StreamingVLM 風に平滑化された安定観測

を管理する。`gradio_demo_*` からは毎フレーム `RawSensorFrame` を `StreamingSensorState` に渡し、そこから得られる `StableSensorSnapshot` を Hub の `MomentLogEntry.metrics` へ反映する。

これにより、MomentLog および HeartOS の memory は

- 瞬間ノイズではなく「継続した情動・場所」のパターン
- 「長く続いた緊張」「特定の場所に長く留まった」などの状態

を正しく捉えられるようになる。Nightly / Monument レイヤでは、`tension_level` や `engagement_level`、`place_confidence` などが「一定しきい値を超えた状態が一定時間以上続いた場合のみ」昇格候補として扱われる。短期的なノイズではなく、時間的に一貫した体験のみが記憶の地形として刻まれる。

さらに、`StableSensorSnapshot.fused_vec` を RWM / Future Replay の観測列 `o_t` として利用することで、

- 終わらないセンサーストリーム
- HeartOS の情動・記憶
- RWM ベースの未来リプレイと予測誤差フィードバック

が単一の観測パイプライン上に統合される。長時間の配信や会議でも、StreamingVLM 風の構造により視覚（webcam）、音声（話者・音量）、テキスト（議事録）を途切れず扱いながら Hub で `sink_vec` を維持し、心拍・集中度・緊張度といった指標を滑らかに HeartOS に渡せる。

結果として、既存の Gradio センサーと HeartOS / Nightly の仕組みをほぼそのまま活かしつつ、「終わらないストリームでも揺らがない心」を持つ EQNet の骨格が成立する。

#### 実装タスクリスト

- **PR-01: StreamingSensorState の追加**
  - `eqnet/hub/streaming_sensor.py` を新設。
  - `RawSensorFrame`, `StableSensorSnapshot`, `StreamingSensorState` を実装。
  - `sink_vec` の EMA 更新、`short_window` の `deque`、`fused_vec = α * sink + (1-α) * window_mean` を実装。
  - `calm_level`, `tension_level`, `engagement_level`, `place_confidence` を返すプレースホルダ関数を用意。

- **PR-02: gradio_demo_* から HubRuntime への配線**
  - `gradio_demo_prev.py` / `_current_copy.py` で mediapipe・音声・webcam の出力を 1 dict にまとめ、`HubRuntime.on_sensor_tick(raw)` を呼ぶ。
  - `eqnet/hub/runtime.py` で `self.streaming_state = StreamingSensorState()` を初期化し、`RawSensorFrame → update() → StableSensorSnapshot → MomentLogEntry.metrics` の流れを作る。

- **PR-03: Nightly / Monument の継続トリガ**
  - `eqnet/heart_os/nightly_*.py` に `MetricEvent` と `promote_monuments_from_metrics()` を追加。
  - `tension > θ_tension` が `min_duration_tension` 秒以上続いた区間や、`engagement > θ_engagement` が一定時間続いた区間のみを Monument 候補として既存昇格処理に渡す。

- **PR-04: RWM / Future Replay との接続**
  - `StableSensorSnapshot.fused_vec` を RWM の観測 `o_t` として渡せるフックを追加。
  - まずはログ保存→オフライン RWM 解析、次の段階でオンライン接続を検討。

### Heartbeat / Monument Handling

- Treat heartbeat or LifeIndicator spikes as *boost factors* rather than standalone triggers.
- Only add monument points when sustained emotional energy (heartbeat + inner speech + pose/voice) and consistent tags (family/place) co-occur.
- Re-evaluate monument candidates during nightly promotion; allow demotion if the context no longer fits.
- Record relationships as contexts (for example, 'time with Ria in the atelier') instead of binding raw physiological readings to a person.
- Provide UX hooks so users can edit or downgrade monuments, keeping the system in line with how families want memories preserved.

This keeps streaming sensors, heartbeats, and inner speech aligned with the shared-memory goals without turning every spike into a permanent monument.


### Heart Rate Decomposition

- Store heart beats as raw physiological values first: heart_rate_raw, short/long HRV, heart_rate_baseline.
- Compute activity_level from pose movement, voice/breath effort, and other motion cues (mediapipe, accelerometer, etc.).
- Derive delta_hr = heart_rate_raw - heart_rate_baseline and split it into:
  - heart_rate_motion when activity_level is high or the rise is slow (walking, exercise).
  - heart_rate_emotion when bctivity_level is low and the rise is fast (inner speech + emotion spike).
  - Mixed cases can be allocated by ratio; only heart_rate_emotion feeds emotional_energy / Monument boosts.
- Keep body_stress_index and autonomic_balance as a separate body-channel; use them to color memories ("body felt tense/overloaded") but never as sole Monument triggers.
- Allow Nightly to adapt heart_rate_baseline so chronic high stress shifts the baseline instead of marking every day as special.

This way, motion-driven heart-rate spikes are recorded without over-weighting them, while quiet emotional spikes still boost monuments and future replay.

#### Heart Rate Decomposition Pseudocode

`
delta_hr = heart_rate_raw - heart_rate_baseline

if activity_level > A_high and delta_hr > HR_th:
    heart_rate_motion  = delta_hr
    heart_rate_emotion = 0.0
elif activity_level < A_low and delta_hr > HR_th:
    heart_rate_motion  = 0.0
    heart_rate_emotion = delta_hr
else:
    alpha = clamp((A_high - activity_level) / (A_high - A_low), 0.0, 1.0)
    heart_rate_motion  = delta_hr * (1 - alpha)
    heart_rate_emotion = delta_hr * alpha
`

Nightly updates the baseline (heart_rate_baseline, body_baseline_shift) slowly so chronic stress shifts the baseline, while monument scoring only inspects heart_rate_emotion. body_stress_index / autonomic_balance remain as body-channel metadata (color / tone) instead of direct triggers; see Monument docs for the scoring reference.

Private flag policy verified

### End-to-End Streaming → Memory → Style Pipeline

- **Sensor layer** (YOLO/MediaPipe/Gaze/HR/VLM) produces `raw_frame` dicts. CPU-only setups can stick to object counts + pose + gaze.
- **StreamingSensorState.from_raw** turns those dicts into a fused observation vector plus structured metrics (`activity_level`, `heart_rate_motion`, `heart_rate_emotion`, `object_counts`, `gaze_vector`, `body_state_flag`, etc.).
- **HubRuntime.on_sensor_tick** stores the latest snapshot and merges the metrics into MomentLog entries so Nightly/Monument/RWM can consume them without re-deriving heuristics.
- **Nightly / Monument** read the emotional metrics while respecting `body_state_flag` (e.g. skip `private_high_arousal`). Repeated, context-consistent events receive Monument points; body metrics remain “color” metadata.
- **Replay / RWM** take `fused_vec` as `o_t` for world-model updates and future replay training.
- **Utterance Style**: emotional/context metrics drive a small `UtteranceStyleState` (pronoun choice, fillers, endings, colloquial intensity, laughter) that post-processes LLM text before TTS/voice, so speech mirrors the sensed body state.

This keeps sensing/physiology, long-term memory, and conversational style connected without blurring module boundaries.

