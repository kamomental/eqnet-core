#!/usr/bin/env bash
set -euo pipefail

# 1) Build candidates
python scripts/build_rag_candidates.py \
  --index rag_index.jsonl \
  --queries queries.jsonl \
  --out out/candidates.rag.jsonl \
  --w_emb 1.0 --topk 20

python scripts/build_etl_candidates.py \
  --episodes logs/episodes.jsonl \
  --out out/candidates.etl.jsonl \
  --beta 0.2 --gamma 0.1

# 2) QUBO selection
SEED=${SEED:-42}
AUDIT=out/audit/qubo_runs.jsonl
python scripts/qubo_select.py \
  --candidates out/candidates.rag.jsonl \
  --mode topk --k 10 --lam 0.25 --mu 0.05 --budget 8 --steps 60000 \
  --seed ${SEED} --auto_temp --early_stop --patience 15000 \
  --out out/selected.rag.json --audit ${AUDIT}

python scripts/qubo_select.py \
  --candidates out/candidates.etl.jsonl \
  --mode budget --budget 50 --lam 0.2 --mu 0.1 --steps 80000 \
  --seed ${SEED} --auto_temp --early_stop --patience 20000 \
  --out out/selected.etl.json --audit ${AUDIT}

# 3) Apply (placeholders)
python scripts/apply_rag_selection.py --selected out/selected.rag.json --apply_log out/apply_rag_log.jsonl
python scripts/apply_etl_selection.py --selected out/selected.etl.json --apply_log out/apply_etl_log.jsonl

# 4) Report
echo "[nightly] generating report..."
python eval/report_nightly.py \
  --current_dir eval/current \
  --report_dir reports

echo "Nightly complete. Audit log:" && tail -n 3 ${AUDIT} || true

# 5) Mood gate A/B (m-ON/OFF, 30 trials)
python ops/eval_mood_ablation.py --trials 30 --out reports/mood_ablation.jsonl || true

echo "--- Replay Memory KPI ---"
python tools/summarize_replay_memory.py | tee -a reports/nightly_metrics.txt || true

echo "--- Replay Memory A/B ---"
python ops/eval_replay_memory_ablation.py --trials 30 --out reports/replay_memory_ablation.json || true

echo "--- Green Response Kernel Suggestion ---"
python tools/tune_green_kernel.py --log logs/decisions.jsonl --out reports/green_kernel_suggestion.json || true

echo "--- Green Response A/B ---"
python ops/eval_green_ablation.py --trials 30 --out reports/green_ablation.json || true
