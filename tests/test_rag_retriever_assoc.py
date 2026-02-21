from __future__ import annotations

import torch

from rag.indexer import IndexedDocument, RagIndex
from rag.retriever import RagRetriever
from runtime.config import RuntimeCfg


def test_retriever_keeps_semantic_order_without_assoc():
    index = RagIndex()
    index.add(
        IndexedDocument(
            doc_id="semantic_top",
            text="a",
            embedding=torch.tensor([1.0, 0.0]),
            metadata={"timestamp_sec": 1},
        )
    )
    index.add(
        IndexedDocument(
            doc_id="semantic_low",
            text="b",
            embedding=torch.tensor([0.8, 0.6]),
            metadata={"timestamp_sec": 100},
        )
    )
    retriever = RagRetriever(index)
    hits = retriever.retrieve(torch.tensor([1.0, 0.0]), top_k=1)
    assert hits[0].doc_id == "semantic_top"


def test_retriever_can_bias_with_temporal_assoc():
    index = RagIndex()
    index.add(
        IndexedDocument(
            doc_id="old_but_semantic",
            text="a",
            embedding=torch.tensor([1.0, 0.0]),
            metadata={"timestamp_sec": 10},
        )
    )
    index.add(
        IndexedDocument(
            doc_id="recent",
            text="b",
            embedding=torch.tensor([0.8, 0.6]),
            metadata={"timestamp_sec": 100},
        )
    )
    retriever = RagRetriever(index)
    hits = retriever.retrieve(
        torch.tensor([1.0, 0.0]),
        top_k=1,
        assoc_context={"timestamp_sec": 100, "temporal_tau_sec": 1.0},
        assoc_weights={
            "semantic": 0.0,
            "temporal": 1.0,
            "affective": 0.0,
            "value": 0.0,
            "open_loop": 0.0,
        },
    )
    assert hits[0].doc_id == "recent"


def test_retrieve_with_assoc_one_liner_path():
    index = RagIndex()
    index.add(
        IndexedDocument(
            doc_id="recent",
            text="a",
            embedding=torch.tensor([1.0, 0.0]),
            metadata={"timestamp_sec": 200},
        )
    )
    index.add(
        IndexedDocument(
            doc_id="old",
            text="b",
            embedding=torch.tensor([0.99, 0.01]),
            metadata={"timestamp_sec": 1},
        )
    )
    retriever = RagRetriever(index)
    cfg = RuntimeCfg()
    cfg.rag.assoc_score.enabled = True
    cfg.rag.assoc_score.temporal_tau_sec = 1.0
    cfg.rag.assoc_score.weights.semantic = 0.0
    cfg.rag.assoc_score.weights.temporal = 1.0
    cfg.rag.assoc_score.weights.affective = 0.0
    cfg.rag.assoc_score.weights.value = 0.0
    cfg.rag.assoc_score.weights.open_loop = 0.0
    hits = retriever.retrieve_with_assoc(
        runtime_cfg=cfg,
        temporal_state={"timestamp_sec": 200},
        query_embedding=torch.tensor([1.0, 0.0]),
        top_k=1,
    )
    assert hits[0].doc_id == "recent"
