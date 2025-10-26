# Next Steps

1. **Configure environment variables**  
   - Adjust fatigue/rest controls in `.env` (`FATIGUE_*`, `AUTO_REST_MODE`, `REST_HISTORY_LIMIT`).  
   - Set `store_diary`/`store_membrane` in consent configs to match privacy requirements.

2. **Run daily job & verify outputs**  
   - Execute `python scripts/run_daily.py ...`.  
   - Inspect `data/state/diary.json`, `rest_state.json`, `field_metrics.json`.

3. **Integrate with your database**  
   - Use `EmotionalMemorySystem.diary_state()` and `rest_state()` to export entries.  
   - Or run `python scripts/export_sqlite.py --state data/state --sqlite diary.db` to create a sample SQLite snapshot (already generated in this session).
   - Build ETL scripts to load diary/rest snapshots into your DB or analytics stack.

4. **Set up dashboards**  
   - Visualise heat metrics, auto-rest history and diary text.  
   - Provide UI toggles allowing users to skip diary entries on heavy days.
   - Use `python scripts/export_timeseries.py --state data/state --out exports/timeseries.csv` for CSV feeds, `python scripts/plot_quicklook.py --state data/state --out figures/sample/quicklook.png` for quick looks, and `python scripts/diary_viewer.py --state data/state` for CLI browsing.

5. **Extend causal analysis**  
   - Add enthalpy and rest-mode indicators to PCMCI/Granger pipelines.  
   - Use `python scripts/granger_analysis.py --csv exports/timeseries.csv --out exports/granger_results.json` to produce baseline causality metrics.
   - For impulse response, run `python scripts/impulse_response.py --csv exports/timeseries.csv --lag 1 --horizon 7 --out exports/irf.json`.
   - For impulse response, run `python scripts/impulse_response.py --csv exports/timeseries.csv --lag 1 --horizon 7 --out exports/irf.json`.
   - Evaluate how interventions (membrane tuning, catalysts) influence fatigue.

6. **Pilot with users**  
   - Collect feedback on diary tone and rest alerts.  
   - Tune fatigue thresholds and diary length to avoid uncanny valley effects.
