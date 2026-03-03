from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from prscope.store import Store


def test_evaluation_deduplication(tmp_path):
    db_path = tmp_path / "prscope.db"
    store = Store(db_path=db_path)

    repo = store.upsert_upstream_repo("owner/repo")
    pr = store.upsert_pull_request(
        repo_id=repo.id,
        number=1,
        state="closed",
        title="Test PR",
        body="",
        author="alice",
        labels=["test"],
        updated_at="2024-01-01T00:00:00Z",
        merged_at="2024-01-02T00:00:00Z",
        head_sha="sha1",
        html_url="https://example.com/pr/1",
    )

    local_profile_sha = "local-sha"
    pr_head_sha = "sha1"

    assert store.evaluation_exists(pr.id, local_profile_sha, pr_head_sha) is False

    store.save_evaluation(
        pr_id=pr.id,
        local_profile_sha=local_profile_sha,
        pr_head_sha=pr_head_sha,
        rule_score=0.8,
        final_score=0.8,
        matched_features=["security"],
        signals={"file_count": 1},
        llm_result=None,
        decision="relevant",
    )

    assert store.evaluation_exists(pr.id, local_profile_sha, pr_head_sha) is True


def test_planning_session_turn_and_version_persistence(tmp_path):
    db_path = tmp_path / "prscope.db"
    store = Store(db_path=db_path)

    session = store.create_planning_session(
        repo_name="alpha",
        title="Test Plan",
        requirements="Build planning mode",
        author_model="gpt-4o-mini",
        critic_model="gpt-4o-mini",
        seed_type="requirements",
    )
    assert session.repo_name == "alpha"
    assert session.current_round == 0
    assert session.author_model == "gpt-4o-mini"
    assert session.critic_model == "gpt-4o-mini"

    turn = store.add_planning_turn(
        session_id=session.id,
        role="critic",
        content="Critique content",
        round_number=1,
        major_issues_remaining=2,
        minor_issues_remaining=1,
        hard_constraint_violations=["C-001"],
    )
    assert turn.major_issues_remaining == 2
    assert turn.hard_constraint_violations == ["C-001"]

    content = "# Plan\n\n- [ ] Item"
    plan_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    version = store.save_plan_version(
        session_id=session.id,
        round_number=1,
        plan_content=content,
        plan_sha=plan_sha,
    )
    assert version.plan_sha == plan_sha

    versions = store.get_plan_versions(session.id, limit=5)
    assert len(versions) == 1
    assert versions[0].round == 1
    assert store.get_latest_critic_turn(session.id) is not None


def test_search_sessions_repo_scope_and_snippet(tmp_path):
    pytest.importorskip("rank_bm25")
    db_path = tmp_path / "prscope.db"
    store = Store(db_path=db_path)

    s1 = store.create_planning_session(
        repo_name="alpha",
        title="Auth Refactor",
        requirements="Refactor authentication flow with token rotation and endpoint updates",
        seed_type="requirements",
    )
    store.save_plan_version(
        session_id=s1.id,
        round_number=1,
        plan_content="A" * 400,
        plan_sha=hashlib.sha256(b"a").hexdigest(),
    )

    s2 = store.create_planning_session(
        repo_name="beta",
        title="Queue Upgrade",
        requirements="Improve queue processing throughput and retries for worker runtime",
        seed_type="requirements",
    )
    store.save_plan_version(
        session_id=s2.id,
        round_number=1,
        plan_content="queue summary",
        plan_sha=hashlib.sha256(b"b").hexdigest(),
    )

    results = store.search_sessions(
        query="refactor authentication flow token rotation endpoint updates",
        repo_name="alpha",
        limit=5,
    )
    assert len(results) == 1
    assert results[0]["session_id"] == s1.id
    assert results[0]["repo_name"] == "alpha"
    assert results[0]["summary_snippet"].endswith("...")


def test_search_sessions_applies_recency_boost(tmp_path):
    pytest.importorskip("rank_bm25")
    db_path = tmp_path / "prscope.db"
    store = Store(db_path=db_path)

    older = store.create_planning_session(
        repo_name="alpha",
        title="Auth Work",
        requirements="Refactor authentication flow token rotation endpoint updates now",
        seed_type="requirements",
    )
    newer = store.create_planning_session(
        repo_name="alpha",
        title="Auth Work",
        requirements="Refactor authentication flow token rotation endpoint updates now",
        seed_type="requirements",
    )
    plan_sha = hashlib.sha256(b"plan").hexdigest()
    store.save_plan_version(older.id, 1, "same content", plan_sha)
    store.save_plan_version(newer.id, 1, "same content", plan_sha)

    with store._connect() as conn:  # noqa: SLF001 - test-only deterministic fixture
        old_date = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
        new_date = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        conn.execute("UPDATE planning_sessions SET created_at = ? WHERE id = ?", (old_date, older.id))
        conn.execute("UPDATE planning_sessions SET created_at = ? WHERE id = ?", (new_date, newer.id))

    results = store.search_sessions(
        query="refactor authentication flow token rotation endpoint updates",
        repo_name="alpha",
        limit=5,
    )
    assert len(results) >= 2
    assert results[0]["session_id"] == newer.id


def test_search_sessions_skips_low_signal_query(tmp_path):
    db_path = tmp_path / "prscope.db"
    store = Store(db_path=db_path)
    store.create_planning_session(
        repo_name="alpha",
        title="Any",
        requirements="Some requirements that exist",
        seed_type="requirements",
    )
    assert store.search_sessions(query="too short query", repo_name="alpha", limit=5) == []


def test_create_planning_session_persists_no_recall(tmp_path):
    db_path = tmp_path / "prscope.db"
    store = Store(db_path=db_path)
    session = store.create_planning_session(
        repo_name="alpha",
        title="No recall plan",
        requirements="Build an alternative architecture with no historical context",
        seed_type="requirements",
        no_recall=True,
    )
    loaded = store.get_planning_session(session.id)
    assert loaded is not None
    assert loaded.no_recall == 1
