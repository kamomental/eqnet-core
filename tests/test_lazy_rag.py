# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import pytest

from emot_terrain_lab.rag.lazy_rag import LazyRAG, LazyRAGConfig
from emot_terrain_lab.hub.llm_hub import LLMHub
from terrain import llm as terrain_llm


def _write_graph(path: Path, node_id: str, title: str) -> None:
    payload = {"nodes": [{"id": node_id, "title": title}], "edges": []}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_lazy_rag_builds_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    graph = tmp_path / "graph.json"
    mem = tmp_path / "logs.jsonl"
    _write_graph(graph, "turn-123", "在庫の差分検知")
    _write_jsonl(
        mem,
        [
            {
                "turn_id": "turn-123",
                "timestamp_ms": 123,
                "text": "棚から部品をピッキングして差分を検知",
            }
        ],
    )
    cfg = LazyRAGConfig(graph_path=graph, memory_jsonl_path=mem, topk_candidates=10, topk_context=3)
    rag = LazyRAG(cfg)
    ctx = rag.build_context("差分検知")
    assert ctx is not None
    assert "参考" in ctx
    assert "差分" in ctx


def test_llmhub_skips_context_when_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph = tmp_path / "graph.json"
    mem = tmp_path / "logs.jsonl"
    _write_graph(graph, "turn-123", "在庫の差分検知")
    _write_jsonl(mem, [{"turn_id": "turn-123", "text": "棚から部品をピッキング"}])

    monkeypatch.setenv("EQNET_LAZY_RAG", "0")
    monkeypatch.setenv("EQNET_LAZY_RAG_GRAPH", str(graph))
    monkeypatch.setenv("EQNET_LAZY_RAG_JSONL", str(mem))

    seen = {}

    def fake_chat(system_prompt: str, prompt: str, **_: object) -> str:
        seen["prompt"] = prompt
        return "ok"

    monkeypatch.setattr(terrain_llm, "chat_text", fake_chat)

    hub = LLMHub()
    hub.generate("差分検知について", context=None, controls={})
    assert seen["prompt"] == "差分検知について"


def test_llmhub_injects_context_when_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph = tmp_path / "graph.json"
    mem = tmp_path / "logs.jsonl"
    _write_graph(graph, "turn-123", "在庫の差分検知")
    _write_jsonl(
        mem,
        [{"turn_id": "turn-123", "timestamp_ms": 123, "text": "棚から部品をピッキング"}],
    )

    monkeypatch.setenv("EQNET_LAZY_RAG", "1")
    monkeypatch.setenv("EQNET_LAZY_RAG_GRAPH", str(graph))
    monkeypatch.setenv("EQNET_LAZY_RAG_JSONL", str(mem))

    seen = {}

    def fake_chat(system_prompt: str, prompt: str, **_: object) -> str:
        seen["prompt"] = prompt
        return "ok"

    monkeypatch.setattr(terrain_llm, "chat_text", fake_chat)

    hub = LLMHub()
    hub.generate("差分検知について", context=None, controls={})
    assert "参考" in seen["prompt"]
    assert "差分検知について" in seen["prompt"]


def test_lazy_rag_returns_none_when_no_matches(tmp_path: Path) -> None:
    graph = tmp_path / "graph.json"
    mem = tmp_path / "logs.jsonl"
    _write_graph(graph, "turn-999", "関係ない話題")
    _write_jsonl(mem, [{"turn_id": "turn-000", "text": "別のログ"}])
    cfg = LazyRAGConfig(graph_path=graph, memory_jsonl_path=mem)
    rag = LazyRAG(cfg)
    assert rag.build_context("差分検知") is None
