from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from prscope.store import Store
from prscope.web.api import RuntimeRegistry, _coerce_stale_processing_payload, create_app


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


def test_reset_clears_visible_session_history(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)
    monkeypatch.setenv("PRSCOPE_CONFIG_ROOT", str(tmp_path))
    store = Store()
    session = store.create_planning_session(
        repo_name=tmp_path.name,
        title="reset me",
        requirements="add health observability",
        seed_type="chat",
        status="refining",
    )
    store.add_planning_turn(session.id, role="user", content="create health api endpoint", round_number=0)
    store.add_planning_turn(session.id, role="author", content="Draft ready.", round_number=0)
    store.save_plan_version(
        session.id,
        round_number=0,
        plan_content="# Old Plan\n\n## Summary\nstale",
        plan_sha="sha-old-plan",
    )
    store.update_planning_session(
        session.id,
        _bypass_protection=True,
        completed_tool_call_groups_json=json.dumps(
            [
                {
                    "sequence": 1,
                    "created_at": "2026-03-06T00:00:00Z",
                    "tools": [{"name": "read_file", "status": "done"}],
                }
            ]
        ),
        diagnostics_json=json.dumps({"warnings_total": 3, "draft_phase": "planner_redraft"}),
    )

    app = create_app()
    client = TestClient(app)

    initial = client.get(f"/api/sessions/{session.id}")
    assert initial.status_code == 200
    assert initial.json()["current_plan"] is not None
    assert initial.json()["conversation"]
    assert initial.json()["session"]["completed_tool_call_groups"]
    assert initial.json()["draft_timing"]["warnings_total"] == 3

    reset = client.post(
        f"/api/sessions/{session.id}/command",
        json={"command": "reset", "command_id": "reset-visible-history", "payload": {}},
    )
    assert reset.status_code == 200

    refreshed = client.get(f"/api/sessions/{session.id}")
    assert refreshed.status_code == 200
    payload = refreshed.json()
    assert payload["session"]["status"] == "draft"
    assert payload["conversation"] == []
    assert payload["plan_versions"] == []
    assert payload["current_plan"] is None
    assert payload["session"]["active_tool_calls"] == []
    assert payload["session"]["completed_tool_call_groups"] == []
    assert payload["draft_timing"]["warnings_total"] == 0
    assert payload["draft_timing"]["draft_phase"] is None


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


def test_list_snapshots_endpoint_returns_items(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)
    monkeypatch.setenv("PRSCOPE_CONFIG_ROOT", str(tmp_path))
    app = create_app()
    client = TestClient(app)

    response = client.get("/api/snapshots")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload.get("items"), list)


def test_get_session_snapshot_returns_payload(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)
    monkeypatch.setenv("PRSCOPE_CONFIG_ROOT", str(tmp_path))
    store = Store()
    session = store.create_planning_session(
        repo_name=tmp_path.name,
        title="snapshot",
        requirements="r",
        seed_type="requirements",
        status="draft",
    )
    snapshot_dir = tmp_path / ".prscope" / "sessions"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / f"{session.id}.json").write_text(
        f'{{"session_id":"{session.id}","updated_at":"2026-03-04T12:00:00Z"}}',
        encoding="utf-8",
    )

    app = create_app()
    client = TestClient(app)
    response = client.get(f"/api/sessions/{session.id}/snapshot")
    assert response.status_code == 200
    payload = response.json()
    assert payload["snapshot"]["session_id"] == session.id


def test_session_diagnostics_uses_persisted_timing(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)
    monkeypatch.setenv("PRSCOPE_CONFIG_ROOT", str(tmp_path))
    store = Store()
    session = store.create_planning_session(
        repo_name=tmp_path.name,
        title="diagnostics",
        requirements="r",
        seed_type="requirements",
        status="draft",
    )
    persisted = {"warnings_total": 3, "author_call_timeouts": 2}
    store.update_planning_session(session.id, diagnostics_json=json.dumps(persisted))

    app = create_app()
    client = TestClient(app)
    response = client.get(f"/api/sessions/{session.id}/diagnostics?repo={tmp_path.name}")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload.get("draft_timing"), dict)
    assert payload.get("draft_timing_source") == "persisted_session"
    assert payload["draft_timing"]["warnings_total"] == 3
    assert payload["draft_timing"]["author_call_timeouts"] == 2


def test_get_session_parses_version_followups_payload(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)
    monkeypatch.setenv("PRSCOPE_CONFIG_ROOT", str(tmp_path))
    store = Store()
    session = store.create_planning_session(
        repo_name=tmp_path.name,
        title="followups",
        requirements="r",
        seed_type="requirements",
        status="refining",
    )
    version = store.save_plan_version(
        session.id,
        round_number=1,
        plan_content="# Plan\n\n## Summary\nready",
        plan_sha="sha-followups",
        followups_json=json.dumps(
            {
                "plan_version_id": 7,
                "questions": [
                    {
                        "id": "response_schema",
                        "question": "Which response shape should we use?",
                        "options": ["preserve existing schema", "add response_time field"],
                        "target_sections": ["architecture"],
                        "concept": "response_schema",
                        "resolved": False,
                    }
                ],
                "suggestions": [{"id": "expand_observability", "suggestion": "Add rollout metrics."}],
            }
        ),
    )
    assert version.id is not None

    app = create_app()
    client = TestClient(app)
    response = client.get(f"/api/sessions/{session.id}")
    assert response.status_code == 200
    payload = response.json()
    current_plan = payload["current_plan"]
    assert current_plan["followups"]["questions"][0]["id"] == "response_schema"
    assert current_plan["followups"]["suggestions"][0]["id"] == "expand_observability"


def test_followup_answer_rejects_stale_plan_version(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)
    monkeypatch.setenv("PRSCOPE_CONFIG_ROOT", str(tmp_path))
    store = Store()
    session = store.create_planning_session(
        repo_name=tmp_path.name,
        title="followups",
        requirements="r",
        seed_type="requirements",
        status="refining",
    )
    version = store.save_plan_version(
        session.id,
        round_number=1,
        plan_content="# Plan\n\n## Summary\nready",
        plan_sha="sha-stale-followups",
        followups_json=json.dumps(
            {
                "plan_version_id": 3,
                "questions": [
                    {
                        "id": "response_schema",
                        "question": "Which response shape should we use?",
                        "options": ["preserve existing schema", "add response_time field"],
                        "target_sections": ["architecture"],
                        "concept": "response_schema",
                        "resolved": False,
                    }
                ],
                "suggestions": [],
            }
        ),
    )
    assert version.id is not None

    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        f"/api/sessions/{session.id}/command",
        json={
            "command": "followup_answer",
            "command_id": "followup-stale",
            "plan_version_id": version.id - 1,
            "followup_id": "response_schema",
            "followup_answer": "preserve existing schema",
        },
    )
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["reason"] == "stale_plan_version"


def test_runtime_registry_tracks_tool_update_and_initial_draft_phase() -> None:
    registry = RuntimeRegistry(emitter=None)  # type: ignore[arg-type]

    registry.note_event(
        "s1",
        {
            "type": "tool_update",
            "tool": {
                "status": "done",
                "durationMs": 12.5,
                "sessionStage": "discovery",
            },
        },
    )
    registry.note_event(
        "s1",
        {
            "type": "phase_timing",
            "session_stage": "initial_draft",
            "state": "start",
        },
    )
    registry.note_event(
        "s1",
        {
            "type": "phase_timing",
            "session_stage": "initial_draft",
            "state": "complete",
            "elapsed_ms": 1500,
            "memory_elapsed_s": 0.2,
            "planner_elapsed_s": 1.3,
            "cache_hit": True,
        },
    )

    timing = registry.get_session_timing("s1")
    assert timing is not None
    assert timing["discovery_tool_calls"] == 1
    assert timing["tool_calls_total"] == 1
    assert timing["initial_draft_started_at_unix_s"] is not None
    assert timing["initial_draft_completed_at_unix_s"] is not None
    assert timing["initial_draft_elapsed_s"] == 1.5
    assert timing["draft_memory_elapsed_s"] == 0.2
    assert timing["draft_planner_elapsed_s"] == 1.3
    assert timing["draft_cache_hit"] is True
    assert timing["first_plan_saved_at_unix_s"] is not None


def test_coerce_stale_payload_moves_done_active_tools_to_completed_groups() -> None:
    class _StoreStub:
        def get_live_running_planning_command(self, _session_id: str) -> None:
            return None

    payload = {
        "status": "draft",
        "phase_message": None,
        "is_processing": False,
        "active_tool_calls": [
            {
                "id": "tool-1",
                "call_id": "tool-1",
                "name": "grep_code",
                "status": "done",
                "created_at": "2026-03-05T20:55:00+00:00",
                "duration_ms": 4,
            }
        ],
        "completed_tool_call_groups": [],
    }

    normalized = _coerce_stale_processing_payload(
        session=SimpleNamespace(id="s1"),
        payload=payload,
        store=_StoreStub(),  # type: ignore[arg-type]
    )

    assert normalized["active_tool_calls"] == []
    assert len(normalized["completed_tool_call_groups"]) == 1
    assert normalized["completed_tool_call_groups"][0]["tools"][0]["name"] == "grep_code"
