from __future__ import annotations

from prscope.model_catalog import get_model, list_models


def test_model_catalog_resolves_known_model():
    model = get_model("gpt-4o-mini")
    assert model is not None
    assert model["provider"] == "openai"

    frontier = get_model("claude-sonnet-4-6")
    assert frontier is not None
    assert frontier["provider"] == "anthropic"


def test_model_catalog_availability_reflects_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    models = list_models()
    openai_model = next(item for item in models if item["model_id"] == "gpt-4o-mini")
    assert openai_model["available"] is False
    assert "missing OPENAI_API_KEY" in str(openai_model.get("unavailable_reason"))

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    refreshed = list_models()
    openai_enabled = next(item for item in refreshed if item["model_id"] == "gpt-4o-mini")
    assert openai_enabled["available"] is True
