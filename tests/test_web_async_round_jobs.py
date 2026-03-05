from __future__ import annotations

import json

from fastapi.testclient import TestClient

from prscope.store import Store
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


def test_round_endpoint_enqueues_job_and_is_idempotent(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)
    monkeypatch.setenv("PRSCOPE_CONFIG_ROOT", str(tmp_path))
    monkeypatch.setenv("PRSCOPE_DISABLE_ROUND_WORKER", "1")
    monkeypatch.setattr("prscope.store.get_prscope_dir", lambda repo_root=None: tmp_path / ".prscope")
    monkeypatch.setattr(
        "prscope.planning.runtime.orchestration.PlanningRuntime.run_adversarial_round",
        lambda self, *args, **kwargs: __import__("asyncio").sleep(0),
    )
    app = create_app()
    client = TestClient(app)

    repo_name = tmp_path.name
    store = Store()
    session = store.create_planning_session(
        repo_name=repo_name,
        title="Queue round",
        requirements="r",
        seed_type="requirements",
        status="refining",
    )

    command_id = "cmd-round-1"
    response = client.post(
        f"/api/sessions/{session.id}/round",
        params={"repo": repo_name},
        json={"command_id": command_id, "user_input": "refine this"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "refining"
    assert "allowed_commands" in payload

    stored_job = store.get_planning_command_by_command_id(session.id, command_id)
    assert stored_job is not None
    assert stored_job.status == "completed"
    assert stored_job.command_id == command_id
    assert stored_job.payload.get("user_input") == "refine this"
    assert stored_job.result_snapshot_json is not None
    parsed_snapshot = json.loads(stored_job.result_snapshot_json)
    assert parsed_snapshot.get("status") == "refining"

    updated_session = store.get_planning_session(session.id)
    assert updated_session is not None
    assert updated_session.current_command_id is None

    # Same command replay is read-only idempotent and returns saved snapshot.
    replay = client.post(
        f"/api/sessions/{session.id}/round",
        params={"repo": repo_name},
        json={"command_id": command_id, "user_input": "refine this"},
    )
    assert replay.status_code == 200
    replay_payload = replay.json()
    assert replay_payload["idempotent_replay"] is True
    assert replay_payload["reason"] == "duplicate_command"
    assert replay_payload["status"] == payload["status"]

    # Different command while an active lease is running is rejected.
    with store._connect() as conn:  # noqa: SLF001
        now = "2999-01-01T00:00:00+00:00"
        conn.execute(
            """
            INSERT INTO planning_commands
                (id, session_id, command, command_id, status, payload_json,
                 result_snapshot_json, started_at, completed_at, last_error,
                 attempt_count, worker_id, lease_expires_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'running', '{}', NULL, ?, NULL, NULL, 0, NULL, ?, ?, ?)
            """,
            ("lock-job", session.id, "run_round", "lock-cmd", now, now, now, now),
        )
    conflict = client.post(
        f"/api/sessions/{session.id}/round",
        params={"repo": repo_name},
        json={"command_id": "cmd-round-2", "user_input": "another"},
    )
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["reason"] == "processing_lock"


def test_command_events_include_command_id_for_distinct_runs(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)
    monkeypatch.setenv("PRSCOPE_CONFIG_ROOT", str(tmp_path))
    monkeypatch.setenv("PRSCOPE_DISABLE_ROUND_WORKER", "1")
    monkeypatch.setattr("prscope.store.get_prscope_dir", lambda repo_root=None: tmp_path / ".prscope")

    emitted_events: list[dict[str, object]] = []

    async def capture_emit(self, session_id: str, event: dict[str, object]) -> None:  # type: ignore[no-untyped-def]
        del self, session_id
        emitted_events.append(dict(event))

    monkeypatch.setattr("prscope.web.events.SessionEventEmitter.emit", capture_emit)

    async def fake_run_adversarial_round(  # type: ignore[no-untyped-def]
        self,
        session_id: str,
        user_input: str | None = None,
        author_model_override: str | None = None,
        critic_model_override: str | None = None,
        event_callback=None,
    ):
        del self, session_id, user_input, author_model_override, critic_model_override
        if event_callback is not None:
            await event_callback({"type": "tool_call", "name": "search_repo", "session_stage": "author"})
            await event_callback(
                {
                    "type": "tool_result",
                    "name": "search_repo",
                    "session_stage": "author",
                    "duration_ms": 7,
                }
            )
            await event_callback({"type": "complete", "message": "Round complete"})
        return None

    monkeypatch.setattr(
        "prscope.planning.runtime.orchestration.PlanningRuntime.run_adversarial_round",
        fake_run_adversarial_round,
    )

    app = create_app()
    client = TestClient(app)

    repo_name = tmp_path.name
    store = Store()
    session = store.create_planning_session(
        repo_name=repo_name,
        title="Run id tagging",
        requirements="r",
        seed_type="requirements",
        status="refining",
    )

    for command_id in ("cmd-run-1", "cmd-run-2"):
        response = client.post(
            f"/api/sessions/{session.id}/command",
            params={"repo": repo_name},
            json={"command": "run_round", "command_id": command_id, "user_input": f"input for {command_id}"},
        )
        assert response.status_code == 200

    tool_calls = [event for event in emitted_events if event.get("type") == "tool_call"]
    tool_results = [event for event in emitted_events if event.get("type") == "tool_result"]
    assert [event.get("command_id") for event in tool_calls] == ["cmd-run-1", "cmd-run-2"]
    assert [event.get("command_id") for event in tool_results] == ["cmd-run-1", "cmd-run-2"]
