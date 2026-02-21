from __future__ import annotations

from typing import Optional

from emot_terrain_lab.hub.llm_hub import LLMHub
from terrain import llm as terrain_llm


class _DummyRAG:
    def __init__(self, *, context: Optional[str], sat_ratio: float) -> None:
        self._context = context
        self.last_score_diag = {"sat_ratio": sat_ratio}

    def build_context(self, user_text: str) -> Optional[str]:
        return self._context


def test_llm_hub_adds_uncertainty_meta_when_low_evidence(monkeypatch):
    monkeypatch.setenv("EQNET_SHOW_UNCERTAINTY_META", "1")
    monkeypatch.setattr(terrain_llm, "chat_text", lambda *args, **kwargs: "ok")
    hub = LLMHub()
    monkeypatch.setattr(hub, "_get_lazy_rag", lambda: _DummyRAG(context=None, sat_ratio=0.85))

    resp = hub.generate("test", context=None, controls={})
    assert resp.confidence < 0.6
    assert "retrieval_sparse" in resp.uncertainty_reason
    assert "score_saturation_high" in resp.uncertainty_reason
    assert "推定信頼度" in resp.text
    assert "（低）" in resp.text
    assert "関連記憶が少ない" in resp.text
    assert "スコア飽和が高い" in resp.text


def test_llm_hub_uncertainty_low_when_context_provided(monkeypatch):
    monkeypatch.setenv("EQNET_SHOW_UNCERTAINTY_META", "1")
    monkeypatch.setattr(terrain_llm, "chat_text", lambda *args, **kwargs: "ok")
    hub = LLMHub()
    resp = hub.generate("test", context="known context", controls={})
    assert resp.confidence >= 0.7
    assert resp.uncertainty_reason == ()
    assert "（中）" in resp.text
    assert "不確実要因: 低" in resp.text


def test_llm_hub_confidence_label_fallback_on_invalid_thresholds(monkeypatch):
    monkeypatch.setenv("EQNET_SHOW_UNCERTAINTY_META", "1")
    monkeypatch.setattr(terrain_llm, "chat_text", lambda *args, **kwargs: "ok")
    hub = LLMHub()
    # invalid order: low >= mid
    hub._runtime_cfg.ui.uncertainty_confidence_low_max = 0.9
    hub._runtime_cfg.ui.uncertainty_confidence_mid_max = 0.7
    resp = hub.generate("test", context="known context", controls={})
    # fallback defaults (0.54, 0.79) should classify ~0.78 as "中"
    assert "（中）" in resp.text


def test_llm_hub_confidence_threshold_fallback_emits_warning(monkeypatch, caplog):
    monkeypatch.setenv("EQNET_SHOW_UNCERTAINTY_META", "1")
    monkeypatch.setattr(terrain_llm, "chat_text", lambda *args, **kwargs: "ok")
    hub = LLMHub()
    hub._runtime_cfg.ui.uncertainty_confidence_low_max = 0.95
    hub._runtime_cfg.ui.uncertainty_confidence_mid_max = 0.70
    caplog.set_level("WARNING")

    _ = hub.generate("test", context="known context", controls={})
    assert "llm_hub.confidence_threshold_fallback" in caplog.text
