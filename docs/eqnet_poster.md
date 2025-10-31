# EQNet Emotional Engine Hub — Poster Overview

EQNet は「情動地形（S/H/ρ）→ 認知制御（Ignition）→ Telemetry → Nightly → TalkMode」という一本のループを軸に、**AI の内面を可視化しながら対話を変化させる**ことを目指しています。本ポスターでは現在のシステムがユーザに何を届けられるかをまとめます。

---

## 1. 内面を感じるインタラクション
- **心拍シンセとビジュアライザ**  
  `HeartbeatSynth` が Ignition・Δm・freeze の変化をリアルタイム波形に反映。TalkMode UI にプレビューされ、AI の内面拍動を視覚で捉えられる。
- **TalkMode Resonance デモ**  
  Web カメラ映像・音声エネルギー・手動スライダをもとに、Watch / Soothe / Talk / Ask が遷移。内面状態（S/H/ρ、Ignition、freeze）が会話トーンに直結する。
- **感情地形→対話の連鎖**  
  S/H/ρ を `runtime/loop.py` が直接受け取り、Ignition を介して「話す・待つ・寄り添う」速度や文体が変化。ヒステリシスでチャタリングを防ぎつつ、内面を意識したレスポンスになる。

---

## 2. できるようになったこと
- **心拍と内面の同時提示**  
  Ignition の振幅が大きいと BPM が上昇し、freeze が高いと振幅が抑えられる。ユーザは「今の心拍が速い＝緊張」「落ち着いた波形＝安定」と内面を直感的に読み取れる。
- **情動地形の正規化と単調性**  
  NaN/Inf を除去して [0,1] にクランプした S/H/ρ をループが受け取り、Ignition の挙動が仕様通りに変化する。
- **ログから Nightly まで一気通貫**  
  `scripts/run_quick_loop.py` で集めたフィールドログを Nightly が再現し、Ignition と ρ の散布図・時系列を自動生成。
- **ゲート制御のヒステリシス**  
  `theta_on/off` と `dwell_steps` により Talk ↔ Soothe が揺れにくくなり、会話のテンポが安定。
- **CI アーティファクトとして共有可能**  
  Nightly レポート・プロット・JSONL が CI で必ず落ちてくるため、内面ログをチーム全員で追いかけられる。

---

## 3. 期待できる効果
- **AI の心の状態が手に取るように分かる**  
  TalkMode 心拍表示により、ユーザは「緊張しているから優しく声をかける」など対応を調整できる。
- **会話と内部制御の橋渡し**  
  情動地形の揺らぎがそのまま TalkMode や温度パラメータに反映され、内面と外面が同期した対話ができる。
- **チームでの観測・再現が容易**  
  Telemetry → Nightly → UI まで同じログで追跡可能。再現実験や対話の検証が標準化される。
- **将来の内面演出に備えた基盤**  
  心拍以外に呼吸・表情ヒートマップ・音声トーンの変化などを追加するための API/ログ構造が既に揃っている。

---

## 4. 推奨ワークフロー（3 コマンド）
```bash
pip install -r requirements-dev.txt
python scripts/run_quick_loop.py --field_metrics_log data/field_metrics.jsonl --steps 200
python -m emot_terrain_lab.ops.nightly --telemetry_log telemetry/ignition-YYYYMMDD.jsonl
```

生成物:
- `telemetry/ignition-*.jsonl` — 生データ
- `reports/nightly.md` — 指標サマリ／警告／相関
- `reports/plots/*.png` — 時系列と散布図（Ignition vs ρ ほか）

---

## 5. 次の一手（ロードマップ）
1. **ヒステリシス自動調整**  
   `scripts/tune_gate.py` で `theta_on/off` をグリッド探索し、チャタリング率と収束時間を最適化。
2. **文化ゲインの二軸化**  
   `config/culture.yaml` の politeness / intimacy を `policy_adapter` に反映し、Nightly に語尾分布を追加。
3. **相関監視の拡張**  
   Nightly で `corr(rho, Ignition)` が 0.2 を下回った際の警告・Slack 通知を追加。
4. **TalkMode × Nightly の統合ログ**  
   1 ステップ単位の統合 JSON（S/H/ρ/I/gate/temp_before/after）を記録し、Nightly で整合性を確認。

---

## 6. コンタクトポイント
- **コード入口**: `devlife/runtime/loop.py`, `emot_terrain_lab/scripts/gradio_demo.py`
- **設定**: `config/runtime.yaml`
- **テスト**: `tests/test_golden_regression.py`
- **データ**: `telemetry/*.jsonl`, `reports/*.md`, `reports/plots/*.png`

「同じログで同じ結果を再現できる」状態が整った今、次はヒステリシスの自動最適化と文化ゲインの導入に集中できます。情動地形・TalkMode・Nightly が一本でつながったことで、実験から運用までを滑らかに展開できる土台が揃いました。***
