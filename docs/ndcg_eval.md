NDCG@10 Evaluation Notes (Δaff RAG)

Goal
- Confirm +5% NDCG@10 after enabling Δaff>τ ETL upserts and MMR retriever.

Procedure
- Run nightly ETL to upsert episodes with |Δaff| > τ into the RAG index.
- Evaluate average NDCG@10 over a held‑out query set with graded relevance labels.

Code Sketch
- Δaff ETL: `rag/aff_etl.py` exposes `upsert_aff_episodes`.
- KPI utilities: `devlife/metrics/kpi.py` exposes `ndcg_at_k` and `ndcg_for_queries`.

Example
```python
from emot_terrain_lab.rag.indexer import RagIndex
from emot_terrain_lab.rag.aff_etl import upsert_aff_episodes, AffEtlConfig
from devlife.metrics.kpi import ndcg_for_queries
import torch

def encode(text: str):
    # placeholder encoder (replace with your embedding model)
    return torch.randn(384)

index = RagIndex()
inserted = upsert_aff_episodes(index, encode, config=AffEtlConfig(tau=0.25))

# suppose we have queries and per‑query relevance maps
rankings = [[("docA", 0.9), ("docB", 0.7), ("docC", 0.6)]]
relevances = [{"docA": 3.0, "docC": 1.0}]
print("NDCG@10:", ndcg_for_queries(rankings, relevances, k=10))
```

Notes
- Keep encoder deterministic for fair comparisons.
- Hold τ fixed across A/B; change only the ETL gating to attribute gains to Δaff.
- Report mean NDCG@10 and standard error across queries.

