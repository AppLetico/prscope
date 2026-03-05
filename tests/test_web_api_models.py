from __future__ import annotations

from fastapi.testclient import TestClient

from prscope.web.api import create_app


def _write_minimal_config(tmp_path):
    (tmp_path / "prscope.yml").write_text(
        "\n".join(
            [
                "local_repo: .",
                "planning:",
                "  author_model: gpt-4o-mini",
                "  critic_model: gpt-4o-mini",
            ]
        ),
        encoding="utf-8",
    )


def test_models_endpoint_returns_catalog(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)
    monkeypatch.setenv("PRSCOPE_CONFIG_ROOT", str(tmp_path))
    app = create_app()
    client = TestClient(app)

    response = client.get("/api/models")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload.get("items"), list)
    assert any(item.get("model_id") == "gpt-4o-mini" for item in payload["items"])
    assert any(item.get("model_id") == "o3" for item in payload["items"])
    assert any(item.get("model_id") == "claude-opus-4-6" for item in payload["items"])
    assert any(item.get("model_id") == "gemini-3.1-pro" for item in payload["items"])


def test_repos_endpoint_returns_configured_profiles(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)
    monkeypatch.setenv("PRSCOPE_CONFIG_ROOT", str(tmp_path))
    app = create_app()
    client = TestClient(app)

    response = client.get("/api/repos")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload.get("items"), list)
    assert len(payload["items"]) >= 1
    assert payload["items"][0]["name"]
    assert payload["items"][0]["path"]


def test_health_endpoint_returns_healthy(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)
    monkeypatch.setenv("PRSCOPE_CONFIG_ROOT", str(tmp_path))
    app = create_app()
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_create_session_rejects_unavailable_model(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)
    monkeypatch.setenv("PRSCOPE_CONFIG_ROOT", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/api/sessions",
        json={
            "mode": "chat",
            "author_model": "gpt-4o-mini",
            "critic_model": "gpt-4o-mini",
        },
    )
    assert response.status_code == 400
    assert "unavailable" in response.json().get("detail", "").lower()
