from scripts.list_llm_models import find_preferred_models


def test_find_preferred_models_matches_case_insensitive_marker() -> None:
    models = [
        {"id": "lmstudio-community/gemma-4-e4b-it"},
        {"id": "qwen3.5-4b"},
        {"id": "unsloth/gemma-4-e4b-it"},
    ]

    assert find_preferred_models(models, "Gemma-4-E4B") == [
        "lmstudio-community/gemma-4-e4b-it",
        "unsloth/gemma-4-e4b-it",
    ]


def test_find_preferred_models_ignores_invalid_entries() -> None:
    models = [
        {"id": ""},
        {},
        {"id": "qwen3.5-4b"},
    ]

    assert find_preferred_models(models, "gemma") == []
