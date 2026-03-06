from __future__ import annotations

import json

import pytest

import prscope.store as store_module
from prscope.store import PRFile, PullRequest, Store


def test_store_package_exports_match_legacy_surface():
    assert Store is not None
    assert PullRequest is not None
    assert PRFile is not None
    assert callable(store_module.get_prscope_dir)


def test_store_default_path_respects_prscope_store_monkeypatch(monkeypatch, tmp_path):
    root = tmp_path / ".prscope"
    monkeypatch.setattr("prscope.store.get_prscope_dir", lambda repo_root=None: root)
    store = Store()
    assert store.db_path == root / "prscope.db"


def test_store_facade_preserves_command_and_session_invariants(tmp_path):
    store = Store(db_path=tmp_path / "test.db")
    session = store.create_planning_session(
        repo_name="repo",
        title="Refactor parity",
        requirements="Keep behavior identical",
        seed_type="requirements",
        status="draft",
    )

    with pytest.raises(RuntimeError):
        store.update_planning_session(session.id, status="refining")

    updated = store.update_planning_session(
        session.id,
        _bypass_protection=True,
        status="refining",
        is_processing=1,
    )
    assert updated.status == "refining"
    assert updated.is_processing == 1

    reserved, replay, conflict, reason = store.reserve_planning_command(
        session_id=session.id,
        command="continue",
        command_id="cmd-1",
        payload_json=json.dumps({"session_id": session.id}),
        lease_seconds=30,
        allowed_commands_by_status={"refining": {"continue"}},
    )
    assert reserved is not None
    assert replay is None
    assert conflict is None
    assert reason is None

    replay_reserved, replay_row, _, replay_reason = store.reserve_planning_command(
        session_id=session.id,
        command="continue",
        command_id="cmd-1",
        payload_json=json.dumps({"session_id": session.id}),
        lease_seconds=30,
        allowed_commands_by_status={"refining": {"continue"}},
    )
    assert replay_reserved is None
    assert replay_row is not None
    assert replay_reason == "duplicate_command"
