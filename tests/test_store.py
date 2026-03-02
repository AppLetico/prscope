from __future__ import annotations

import hashlib

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
