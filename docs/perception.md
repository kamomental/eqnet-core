# Perception Bridge Guide

EQNetの Perception Bridge は、Webカメラ／マイク、さらにはロボットからのセンサー入力を「感情の火種 (valence/arousal)」へ変換する層です。リアルタイム用途と夜間バッチ処理の両方を想定し、MediaPipe / OpenCV / ONNX などへの差し替えが可能なプラガブル設計

---

## 1. 設定
`hub.PerceptionConfig` の主な項目:

```yaml
perception:
  video_fps: 30
  downscale: 256
  audio_sample_rate: 16000
  fusion_hz: 15
  video_backend: "opencv"      # "opencv" | "mediapipe" | "onnx" | custom
  video_backend_params:
    model_path: null           # ONNXモデルなど
  audio_backend: "lightweight" # "lightweight" | "webrtcvad" |  custom
  audio_backend_params:
    vad_mode: 2
  batch_enabled: true
```

### Video backends
- **opencv**: 既存ヒューリスティック。どの環境でも動作。  
- **mediapipe**: 顔ランドマークを利用して口開き/瞬目等を抽出（MediaPipeがインストールされている場合）。  
- **onnx**: 任意の ONNX モデルを読み込み。`model_path` を設定して利用。

### Audio backends
- **lightweight**: RMS + Zero-Crossing Rate（既定）。  
- **webrtcvad**: WebRTC VAD で音声区間を判定（`vad_mode` 0〜3）。

---

## 2. リアルタイムとバッチ
- リアルタイム：`observe_video(frame)` / `observe_audio(chunk)` をWebカメラ・マイクから周期的に呼び出す。  
- バッチ（夜間処理など）：`process_video_batch(frames)` / `process_audio_batch(chunks)` を利用。`batch_enabled` が false の場合は使用できません。

---

## 3. クロスプラットフォーム対応
- **Windows**: OpenCV / MediaPipe / PyAudio を標準想定。  
- **Linux / macOS**: 同上、ROS2 や Isaac Sim と組み合わせる場合も `perception` 層は共通。  
- **モバイル**: MediaPipe のモバイルランタイムや端末独自のカメラAPIを利用。`video_backend` をカスタムクラスに差し替えることで適合可能。

---

## 4. 拡張ポイント
- 新しいバックエンドを追加する場合は `hub/perception/video_backends.py` / `audio_backends.py` にクラスを定義し、`create_*_backend()` に登録。  
- ロボットセンサー（LiDAR/IMU など）を組み込みたい場合は、別モジュールで解析後に `AffectSample` の補助情報として利用してください。

---

リアルタイム運用も夜間バッチも、このPerception Bridgeが「語る前に感じる」EQNetの最初の火種です。必要に応じてプラグインを追加し、SLO（p95 ≤ 50ms）を意識しながらチューニングを行ってください。
