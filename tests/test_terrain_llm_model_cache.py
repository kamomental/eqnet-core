from pathlib import Path

from emot_terrain_lab.terrain import llm


class _DummyClient:
    pass


def test_pick_model_prefers_cached_selected_model() -> None:
    chosen = llm.pick_model(
        ["qwen-3.5-instruct", "phi-4-mini", "llama-3.1"],
        cached_selected="phi-4-mini",
    )

    assert chosen == "phi-4-mini"


def test_resolve_endpoint_falls_back_to_cached_model_list(monkeypatch, tmp_path: Path) -> None:
    cache_path = tmp_path / "llm_model_cache.json"
    monkeypatch.setattr(llm, "LLM_MODEL_CACHE_PATH", str(cache_path))
    monkeypatch.setattr(llm, "CUSTOM_BASE", None)
    monkeypatch.setattr(llm, "CUSTOM_KEY", None)
    monkeypatch.setattr(llm, "CUSTOM_MODEL", None)
    monkeypatch.setattr(llm, "LM_BASE", "http://localhost:1234/v1")
    monkeypatch.setattr(llm, "LM_KEY", "lm-studio")
    monkeypatch.setattr(llm, "LM_MODEL", None)
    monkeypatch.setattr(llm, "_client", lambda base, key: _DummyClient())
    monkeypatch.setattr(llm, "list_models", lambda base, key: [])

    llm._store_model_cache(
        "http://localhost:1234/v1",
        ["qwen-3.5-instruct", "phi-4-mini"],
        "phi-4-mini",
    )

    info = llm._resolve_endpoint()

    assert info is not None
    assert info.model == "phi-4-mini"
    assert info.model_source == "cache"


def test_resolve_endpoint_updates_cache_from_live_model_list(monkeypatch, tmp_path: Path) -> None:
    cache_path = tmp_path / "llm_model_cache.json"
    monkeypatch.setattr(llm, "LLM_MODEL_CACHE_PATH", str(cache_path))
    monkeypatch.setattr(llm, "CUSTOM_BASE", None)
    monkeypatch.setattr(llm, "CUSTOM_KEY", None)
    monkeypatch.setattr(llm, "CUSTOM_MODEL", None)
    monkeypatch.setattr(llm, "LM_BASE", "http://localhost:1234/v1")
    monkeypatch.setattr(llm, "LM_KEY", "lm-studio")
    monkeypatch.setattr(llm, "LM_MODEL", None)
    monkeypatch.setattr(llm, "_client", lambda base, key: _DummyClient())
    monkeypatch.setattr(
        llm,
        "list_models",
        lambda base, key: ["phi-4-mini", "qwen-3.5-instruct", "llama-3.1"],
    )

    info = llm._resolve_endpoint()
    cache = llm._load_model_cache()
    entry = cache.get("http://localhost:1234/v1") or {}

    assert info is not None
    assert info.model == "qwen-3.5-instruct"
    assert info.model_source == "live_list"
    assert entry.get("selected_model") == "qwen-3.5-instruct"
    assert "qwen-3.5-instruct" in (entry.get("models") or [])


def test_prefer_cached_model_promotes_target_model_without_live_list(monkeypatch, tmp_path: Path) -> None:
    cache_path = tmp_path / "llm_model_cache.json"
    monkeypatch.setattr(llm, "LLM_MODEL_CACHE_PATH", str(cache_path))

    llm.prefer_cached_model(
        "http://localhost:1234/v1",
        "qwen-3.5-coder",
        available_models=["phi-4-mini"],
    )

    cache = llm._load_model_cache()
    entry = cache.get("http://localhost:1234/v1") or {}

    assert entry.get("selected_model") == "qwen-3.5-coder"
    assert entry.get("models", [])[0] == "qwen-3.5-coder"
    assert llm.get_cached_selected_model("http://localhost:1234/v1") == "qwen-3.5-coder"
