# Emotional Terrain Lab

A toolkit for experimenting with **multi-layer emotional memory**, **thermodynamic field dynamics** (entropy/enthalpy), **catalyst events**, and **reflective daily diaries**.  
Runs locally with minimal setup. Produces dashboards, diary logs, and causal analysis artefacts.

“This document outlines only the concept, as the work period is limited to October 20–26, 2025.”
---

## 🚀 Quick Start

### 1. Install Python
Install **Python 3.11+** from [python.org](https://www.python.org/downloads/).

### 2. Run Everything Automatically
- **Windows:** Double-click `quickstart.bat`  
- **macOS/Linux:** Run `./quickstart.sh` (make it executable with `chmod +x quickstart.sh`)

This helper script will:
- Generate sample conversations (if none exist)
- Execute daily + weekly pipelines (memory updates, diary creation, rest detection)
- Output:
  - `diary_quickstart.db`
  - `exports/timeseries_quickstart.csv`
  - `exports/granger_quickstart.json`
  - `exports/irf_quickstart.json`
  - `figures/sample/quicklook.png`

### 3. Review Results
```bash
python scripts/diary_viewer.py --state data/state
```
- Quicklook chart: `figures/sample/quicklook.png`
- CSV/JSON: `exports/` directory

---

## 🧩 Audit & Nightly Helpers

- `quickstart_audit.bat` / `.sh`: lightweight audit (fast-path validation + nightly aggregation)
- Recommended CI order:
  ```
  quickstart.bat → quickstart_audit.bat
  ```
- For a full nightly run, use `ops/nightly.sh`.

---

## 💾 Large Data Assets

- `emot_terrain_lab/data/` and root `data/` may hold 10–60 GB of `.jsonl` or state files.  
- These are `.gitignore`d on purpose — **do not commit**.  
- Store in S3/GCS/internal storage and document sync steps in `docs/terrain_pipeline.md`.

---

## 🖥️ Local Stack Launcher

- Windows: `start_local_stack.bat`  
- PowerShell (cross-platform):  
  ```bash
  pwsh -File emot_terrain_lab/scripts/start_local_stack.ps1
  ```

### Components Launched
1. `ops/hub_ws.py` — WebSocket bus  
2. `ops/config_watcher.py` — YAML watcher (Ctrl+R/SIGHUP reload)  
3. `ops/hotkeys.py` — Hotkeys (F9/F10/F11)  
4. `ops/dashboard.py` — Sigma/Psi dashboard (http://127.0.0.1:8080)

Options: `-NoWatcher`, `-NoHotkeys`, `-NoDashboard`, `-LogLevel DEBUG`  
If `.venv` exists, it will auto-activate.

---

## ⚙️ Environment Setup

```bash
python -m venv .venv
. .venv/Scripts/activate      # Windows
source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

Optional: enable LM Studio (`USE_LLM=1` in `.env`, default endpoint `http://localhost:1234/v1`).

---

## 🔄 Core Workflow

```bash
python scripts/simulate_sessions.py --users 3 --weeks 4 --out data/logs.jsonl
python scripts/run_daily.py --in data/logs.jsonl --state data/state --user user_000
python scripts/run_weekly.py --state data/state
python scripts/predict_next_week.py --state data/state --in data/logs.jsonl --out data/preds.csv --user user_000
python scripts/visualize.py --state data/state --in data/logs.jsonl --out figures --user user_000
python scripts/diary_viewer.py --state data/state
python scripts/export_sqlite.py --state data/state --sqlite diary.db
python scripts/export_timeseries.py --state data/state --out exports/timeseries.csv
python scripts/granger_analysis.py --csv exports/timeseries.csv --out exports/granger_results.json
python scripts/impulse_response.py --csv exports/timeseries.csv --lag 1 --horizon 7 --out exports/irf.json
python scripts/plot_quicklook.py --state data/state --out figures/sample/quicklook.png
```

---

## 📂 Key Artefacts

| File | Description |
|------|--------------|
| `data/state/diary.json` | Diary entries (respecting `store_diary`) |
| `data/state/rest_state.json` | Auto-rest activation log |
| `exports/timeseries.csv` | Entropy/enthalpy + rest flags |
| `exports/granger_results.json` | Granger causality p-values |
| `exports/irf.json` | VAR impulse-response data |
| `figures/sample/quicklook.png` | Quicklook chart |
| `diary.db` | SQLite snapshot (for BI tools) |

---

## 🧠 Script Reference

| Script | Description |
|---------|-------------|
| `scripts/run_daily.py` | Daily pipeline |
| `scripts/run_weekly.py` | Weekly abstraction |
| `scripts/export_sqlite.py` | Export diary/rest to SQLite |
| `scripts/export_timeseries.py` | CSV export |
| `scripts/granger_analysis.py` | Granger causality tests |
| `scripts/impulse_response.py` | VAR shock analysis |
| `scripts/plot_quicklook.py` | Quick entropy/enthalpy plot |
| `scripts/diary_viewer.py` | Text diary browser |
| `scripts/update_community_terms.py` | Maintain slang dictionary |
| `scripts/harvest_neologisms.py` | Extract neologisms |
| `scripts/demo_hub.py` | EQNet Hub demo |

---

## ⚙️ Configuration Tips

- `.env`: thresholds, auto-rest, diary storage, LM Studio endpoint  
- `config/culture.yaml`: cultural projection matrices  
- `config/prosody.yaml`: prosody-to-emotion blending  
- `config/dream.yaml`: DreamLink (G2L + RAE) settings  
- `config/hub.yaml`: LLM routing / `config/tools.yaml`: tool registry  
- `config/robot.yaml`: ROS2/Isaac bridge toggles  
- `resources/community_terms.yaml`: slang by period  
- `resources/community_reply_templates.yaml`: reply templates  
- `ENABLE_COMMUNITY_ORCHESTRATOR=1`: enable multi-speaker coordinator  
- `data/logs*.jsonl`: conversation logs (replace simulator output)

---

## 📦 Requirements

Core dependencies:
- NumPy, Pandas, Matplotlib  
- scikit-learn, statsmodels  
- textual, PyYAML, Torch (CPU by default)  
- OpenAI client (for LM Studio)

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🪜 Next Steps

Refer to `NEXT_STEPS.md` for:
- Environment tuning  
- Dashboard integration  
- Extended analytics (Granger/IRF)  
- Diary reviewer pipelines  

---

## 🪄 Local Stack Launcher (Summary)

- **Windows:** `start_local_stack.bat`  
- **PowerShell (cross-platform):**
  ```bash
  pwsh -File emot_terrain_lab/scripts/start_local_stack.ps1
  ```
- Flags: `-NoWatcher`, `-NoHotkeys`, `-NoDashboard`
- Auto-activates `.venv` if found
