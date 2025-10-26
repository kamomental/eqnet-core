from emot_terrain_lab.hub.hub import Hub


class StubModel:
    def generate_text(self, prompt: str, **_: float) -> str:
        return f"echo:{prompt}"


def test_engine_slot_transformer_stub():
    cfg = {
        "hub": {"inference": "transformer", "mode": "Supportive", "heartiness": 0.2},
        "transformer_model": StubModel(),
        "transformer_tokenizer": None,
        "green": {"culture_resonance": 0.3},
    }
    hub = Hub(cfg)
    result = hub.run({"prompt": "hello", "style": "chat_support"})
    receipt = result["receipt"]
    assert result["text"].startswith("echo:")
    assert "engine_trace" in receipt
    assert receipt["action"] == "explain_options"
    assert receipt["norms"]["politeness"] == 0.6


def test_engine_slot_bdh_stub():
    cfg = {
        "hub": {"inference": "bdh", "mode": "Reflective", "heartiness": 0.6},
        "inference": {"bdh": {"model_path": "models/bdh-small", "device": "cpu"}},
        "green": {"culture_resonance": 0.3},
    }
    hub = Hub(cfg)
    result = hub.run(
        {
            "prompt": "hello",
            "style": "tidy_humming",
            "qualia": {"tone": "soft"},
        }
    )
    receipt = result["receipt"]
    assert receipt["engine_trace"]["engine"] == "bdh"
    assert receipt["action"] in {"explain_options", "ask_consent"}
