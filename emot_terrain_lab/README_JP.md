# Emotional Terrain Lab（感情地形ラボ）

マルチレイヤの感情記憶、熱力学的フィールド（エントロピー／エンタルピー）、
触媒イベント、日次リフレクション日記を扱う実験用ツールキット。  
ローカル環境で簡単に実行でき、ダッシュボード・日記ログ・因果解析レポートを生成します。

(作業期間 2025/10/20-10/26のためコンセプトのみ)
---

## 🚀 クイックスタート

### 1. Pythonの準備
[python.org](https://www.python.org/downloads/) から **Python 3.11+** をインストール。

### 2. 自動実行
- **Windows:** `quickstart.bat` をダブルクリック  
- **macOS/Linux:** `./quickstart.sh` （最初に `chmod +x quickstart.sh`）

このスクリプトは以下を自動実行します：
- サンプル会話を生成（存在しない場合）
- 日次＋週次パイプラインを処理（記憶更新／日記生成／休息検出）
- 出力：
  - `diary_quickstart.db`
  - `exports/timeseries_quickstart.csv`
  - `exports/granger_quickstart.json`
  - `exports/irf_quickstart.json`
  - `figures/sample/quicklook.png`

### 3. 結果確認
- 対話ビューア  
  ```bash
  python scripts/diary_viewer.py --state data/state
  ```
- Quicklook 図  
  `figures/sample/quicklook.png`
- CSV/JSON レポート  
  `exports/` ディレクトリを参照

---

## 🧩 Audit & Nightly Helpers

- `quickstart_audit.bat` / `.sh`: 軽量監査（fast-path設定＋Nightly集計）
- CIやcronでは：
  ```
  quickstart.bat → quickstart_audit.bat
  ```
- フルNightlyは `ops/nightly.sh` を実行。

---

## 💾 大容量データの扱い

- `emot_terrain_lab/data/` や `data/` に10〜60GBのstateやjsonlが生成されます。  
  `.gitignore`済なのでGitに含めないでください。
- 代わりに S3 / GCS / ストレージに保存。
- `docs/terrain_pipeline.md` に同期手順を記載。

---

## 🖥️ ローカルスタック起動

- Windows: `start_local_stack.bat`
- macOS/Linux(PowerShell):  
  ```bash
  pwsh -File emot_terrain_lab/scripts/start_local_stack.ps1
  ```

### 起動されるプロセス
1. `ops/hub_ws.py` — WebSocketバス  
2. `ops/config_watcher.py` — YAML変更監視  
3. `ops/hotkeys.py` — ホットキー（F9/F10/F11）  
4. `ops/dashboard.py` — ダッシュボード (http://127.0.0.1:8080)

オプション: `-NoWatcher`, `-NoHotkeys`, `-NoDashboard`, `-LogLevel DEBUG`  
`.venv` があれば自動有効化。

---

## ⚙️ 環境構築

```bash
python -m venv .venv
. .venv/Scripts/activate      # Windows
source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

LM Studioを使う場合は `.env` の `USE_LLM=1` を有効化。  
デフォルトエンドポイントは `http://localhost:1234/v1`。

---

## 🔄 コアワークフロー

```bash
# 会話シミュレーション（任意）
python scripts/simulate_sessions.py --users 3 --weeks 4 --out data/logs.jsonl

# 日次統合＋日記生成（L1→L2）
python scripts/run_daily.py --in data/logs.jsonl --state data/state --user user_000

# 週次抽象化（L2→L3）
python scripts/run_weekly.py --state data/state

# 予測デモ（任意）
python scripts/predict_next_week.py --state data/state --in data/logs.jsonl --out data/preds.csv --user user_000

# 可視化
python scripts/visualize.py --state data/state --in data/logs.jsonl --out figures --user user_000

# 日記ビューア
python scripts/diary_viewer.py --state data/state

# データエクスポート
python scripts/export_sqlite.py --state data/state --sqlite diary.db
python scripts/export_timeseries.py --state data/state --out exports/timeseries.csv
python scripts/granger_analysis.py --csv exports/timeseries.csv --out exports/granger_results.json
python scripts/impulse_response.py --csv exports/timeseries.csv --lag 1 --horizon 7 --out exports/irf.json
python scripts/plot_quicklook.py --state data/state --out figures/sample/quicklook.png
```

---

## 📂 主要ファイル

| ファイル | 内容 |
|-----------|------|
| `data/state/diary.json` | 日記エントリ（`store_diary` 設定に従う） |
| `data/state/rest_state.json` | 自動休息履歴 |
| `exports/timeseries.csv` | 熱力学指標＋休息フラグ |
| `exports/granger_results.json` | グレンジャー因果性 |
| `exports/irf.json` | VARインパルス応答 |
| `diary.db` | SQLiteスナップショット |
| `figures/sample/quicklook.png` | QuickLookチャート |

---

## 🧠 スクリプトリファレンス

| スクリプト | 機能 |
|-------------|------|
| `scripts/run_daily.py` | 日次パイプライン |
| `scripts/run_weekly.py` | 週次抽象化 |
| `scripts/export_sqlite.py` | SQLite出力 |
| `scripts/export_timeseries.py` | CSVエクスポート |
| `scripts/granger_analysis.py` | 因果性テスト |
| `scripts/impulse_response.py` | ショック解析 |
| `scripts/plot_quicklook.py` | 熱力学グラフ |
| `scripts/diary_viewer.py` | 日記ブラウザ |
| `scripts/update_community_terms.py` | スラング辞書更新 |
| `scripts/harvest_neologisms.py` | 新語抽出 |
| `scripts/demo_hub.py` | EQNet Hubデモ |

---

## ⚙️ 設定ファイルのヒント

- `.env` — 閾値、休息、自動処理、LM Studio設定  
- `config/culture.yaml` — 文化的投影行列  
- `config/prosody.yaml` — 音声抑揚→感情ブレンド設定  
- `config/dream.yaml` — DreamLink(G2L+RAE)設定  
- `config/hub.yaml` — LLMルーティング  
- `config/robot.yaml` — ロボットブリッジ設定  
- `resources/community_terms.yaml` — スラング辞書  
- `resources/community_reply_templates.yaml` — 返信テンプレート  
- `ENABLE_COMMUNITY_ORCHESTRATOR=1` で複数話者調整を有効化  
- `data/logs*.jsonl` — 会話ログ（1行1イベント）

---

## 📦 依存ライブラリ

- NumPy  
- Pandas  
- Matplotlib  
- scikit-learn  
- statsmodels  
- textual  
- PyYAML  
- Torch（CPUデフォルト）  
- OpenAI client（LM Studio互換）

---

## 🪜 次のステップ

- `NEXT_STEPS.md` を確認  
  - 環境調整  
  - ダッシュボード統合  
  - Granger/IRF 分析拡張  
  - 日記レビュア機能 など

---

## 🪄 ローカルスタック補足

- Windows: `start_local_stack.bat`  
- PowerShell:  
  ```bash
  pwsh -File emot_terrain_lab/scripts/start_local_stack.ps1
  ```
- フラグ: `-NoWatcher`, `-NoHotkeys`, `-NoDashboard`  
- `.venv` 自動有効化対応



