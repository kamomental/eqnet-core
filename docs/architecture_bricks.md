# EQNet Brick Architecture（2025-10 更新）

本ドキュメントは EQNet の主要ブリックを「データの流れ」と「責務」で整理するアーキテクチャ・ポスターです。最新の改修で、情動地形 → ランタイム → Telemetry → Nightly → TalkMode までが一貫して動作するようになりました。

---

## 1. Signal Families（シグナル種別）

| Symbol | Type                         | 説明                                                                 |
| ------ | ---------------------------- | -------------------------------------------------------------------- |
| **S1** | `timeseries<float>@1Hz`      | センサ起源の連続値（S/H/ρ、Ignition、freeze など）。1Hz サンプリング。 |
| **S2** | `event<int>@1Hz`             | 離散イベント（bud、policy action、gate state）。                      |
| **S3** | `field<float>[64x64]`        | Green カーネルによるフィールドマップ。64×64 のグリッドを想定。       |
| **S4** | `log: jsonl`                 | Telemetry／Nightly が扱う JSONL ログ。                              |
| **S5** | `artifact: png/md`           | 可視化アーティファクト（plots/*.png、reports/nightly.md）。        |

---

## 2. Brick Catalog（モジュール一覧）

| Brick        | Inputs                               | Outputs                    | Purpose                                               | Module / Entrypoint                                      |
| ------------ | ------------------------------------ | -------------------------- | ----------------------------------------------------- | -------------------------------------------------------- |
| **SENSE**    | sensors                              | S1 (raw)                   | センサ前処理・正規化・クリッピング                    | `terrain/system.py`, `ingest/*`                          |
| **FIELD**    | S1 (bud, weights)                    | S1 (S/H/ρ), S3 (field)     | Green Kernel による情動地形の更新                     | `terrain/field.py`, `tools/tune_green_kernel.py`         |
| **PHASE**    | S1 (S/H/ρ)                           | S1 (Ignition, Δaff)        | spec 準拠の Ignition 計算・単調性チェック             | `devlife/runtime/loop.py`                                |
| **POLICY**   | S1 (Ignition, mood), S2 (bud)        | S2 (action)                | policy_adapter による温度・待ち時間のゲイン計算       | `policy_adapter.py`, `runtime/config.py`                 |
| **SAFETY**   | S2 (action), S1 (rho, freeze)        | S2 (guarded action)        | 符号と飽和のゲート制御（ヒステリシス）                | `devlife/runtime/loop.py`                                |
| **TALKMODE** | S1 (Ignition, freeze), S2 (gate)     | UI (Gradio)                | Talk/Watch/Soothe のリアルタイム切替と可視化           | `emot_terrain_lab/scripts/gradio_demo.py`                |
| **TELEMETRY**| S1/S2                                | S4 (jsonl)                 | 連続ログ出力・スキーマ付与                            | `telemetry/*`, `devlife/runtime/loop.py`                 |
| **NIGHTLY**  | S4 (jsonl), S5 (plots)               | S5 (png, md)               | 集計・散布図・警告生成                                | `ops/nightly.py`, `telemetry/plot_ignition.py`           |
| **TESTS**    | fixtures                             | assertions                 | ゴールデンログ回帰、ヒステリシスの回数減確認          | `tests/test_golden_regression.py`, `tests/test_gate_*`   |

---

## 3. Wiring Rules（配線ルール）

1. **SENSE → FIELD → PHASE**  
   `tools/tune_green_kernel.py` で決めたクランプを適用し、S/H/ρ が [0,1] でループへ渡ることを保証する。
2. **PHASE → POLICY / SAFETY**  
   Ignition のヒステリシス閾値 (`theta_on/off`) と `dwell_steps` は `config/runtime.yaml` で集中管理。ゲイン符号は `policy_adapter.py` の一箇所で定義。
3. **PHASE / SAFETY → TALKMODE**  
   TalkMode UI は S1 (Ignition, freeze) と gate state を受け取り、チャタリングを抑えた表示を行う。
4. **PHASE → TELEMETRY → NIGHTLY**  
   Telemetry は schema 付き JSONL (`schema: field.v1`) を書き出す。Nightly はそのまま読み、散布図と時系列プロットを生成。
5. **NIGHTLY ↔ TESTS**  
   `tests/fixtures/golden/field_metrics.jsonl` を使って平均 Ignition と Corr(ρ,I) を閾値監視。Nightly の結果とズレたらテストが失敗する。

---

## 4. Reference Build（再現可能な配線例）

```yaml
build: Field-Ignition Loop
wires:
  - SENSE.raw             -> FIELD.input
  - FIELD.metrics         -> PHASE.input
  - PHASE.ignition        -> POLICY.ctx
  - PHASE.ignition        -> SAFETY.ctx
  - SAFETY.gate_state     -> TALKMODE.ui
  - PHASE.metrics         -> TELEMETRY.log
  - TELEMETRY.log         -> NIGHTLY.report
guards:
  theta_on: 0.62
  theta_off: 0.48
  dwell_steps: 8
artifacts:
  - telemetry/ignition-*.jsonl
  - reports/nightly.md
  - reports/plots/*.png
```

---

## 5. 運用チェックリスト
- [ ] `python scripts/run_quick_loop.py --field_metrics_log …` でフィールドログを再生できる  
- [ ] `python -m emot_terrain_lab.ops.nightly --telemetry_log …` が成功し、散布図・Markdown が生成される  
- [ ] `tests/test_golden_regression.py` が緑（平均 Ignition と Corr(ρ,I) が基準内）  
- [ ] Gradio TalkMode の Talk/Watch/Soothe がヒステリシス通りに切り替わる  
- [ ] CI アーティファクトに `reports/nightly.md`・`reports/plots/*.png`・`telemetry/*.jsonl` が含まれる

---

## 6. 今後の拡張ポイント
1. **ゲイン自動探索** — `scripts/tune_gate.py` で theta_on/off を最適化し、Nightly のチャタリング率を自動レポート。  
2. **文化ゲインの導入** — politeness / intimacy 軸を `policy_adapter.py` に接続し、Nightly へ語尾分布を追加。  
3. **Telemetry スキーマ v2** — Gate 状態・温度制御の符号一致率を JSONL に追加し、Nightly で監視。  
4. **Granger / Hawkes 解析** — `ops/hawkes_light.py` に S/H/ρ → Ignition の因果モードを追加し、Nightly に要約を表示。  
5. **UI と Nightly のリンク** — TalkMode で再生したシナリオが Nightly 散布図から参照できるよう統合 ID を発行。

---

このアーキテクチャ・ブリックは、実験から運用までを同じログで追跡するための共通言語です。ブリック間の配線とガード値を明示しておくことで、新しいモジュールが増えても「どこに接続すればよいか」「どの指標を監視すべきか」が一目でわかります。 README のクイックスタート手順と併せて活用してください。***
