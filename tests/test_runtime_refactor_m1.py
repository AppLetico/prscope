from __future__ import annotations

import json

import pytest

from prscope.config import IssueDedupeConfig, PlanningConfig, PrscopeConfig, RepoProfile
from prscope.planning.runtime.critic import ImplementabilityResult, ReviewResult
from prscope.planning.runtime.orchestration import PlanningRuntime
from prscope.planning.runtime.pipeline.round_context import PlanningRoundContext
from prscope.planning.runtime.review.issue_similarity import IssueSimilarityService
from prscope.store import Store


def _runtime(tmp_path, *, planning: PlanningConfig | None = None) -> PlanningRuntime:
    store = Store(tmp_path / "test.db")
    config = PrscopeConfig(
        local_repo=str(tmp_path),
        planning=planning or PlanningConfig(author_model="gpt-4o-mini", critic_model="gpt-4o-mini"),
    )
    repo = RepoProfile(name="repo", path=str(tmp_path))
    return PlanningRuntime(store=store, config=config, repo=repo)


def test_orchestration_support_helpers_are_wired(tmp_path):
    runtime = _runtime(tmp_path)
    assert runtime._event_router is not None  # noqa: SLF001
    assert runtime._snapshot_io is not None  # noqa: SLF001
    assert runtime._initial_draft is not None  # noqa: SLF001
    assert runtime._session_starts is not None  # noqa: SLF001
    assert runtime._chat_flow is not None  # noqa: SLF001
    assert runtime._round_entry is not None  # noqa: SLF001


def test_state_bootstrap_is_lazy_for_repo_memory(tmp_path):
    runtime = _runtime(tmp_path)
    session = runtime.store.create_planning_session(
        repo_name="repo",
        title="lazy",
        requirements="r",
        seed_type="requirements",
        status="draft",
    )
    call_count = {"n": 0}

    def fake_load_all_blocks():
        call_count["n"] += 1
        return {"architecture": "arch"}

    runtime.memory.load_all_blocks = fake_load_all_blocks  # type: ignore[method-assign]
    state = runtime._state(session.id, session)  # noqa: SLF001

    assert call_count["n"] == 0
    assert state.repo_memory == {}

    blocks = runtime._repo_memory(state)  # noqa: SLF001
    assert call_count["n"] == 1
    assert blocks["architecture"] == "arch"


def test_working_summary_falls_back_when_compressor_errors(tmp_path):
    runtime = _runtime(tmp_path)

    class BrokenCompressor:
        def summarize(self, rounds):  # type: ignore[no-untyped-def]
            del rounds
            raise RuntimeError("boom")

    runtime._compressor = BrokenCompressor()  # type: ignore[assignment]  # noqa: SLF001

    summary = runtime._build_working_summary(  # noqa: SLF001
        requirements="Refine architecture",
        critic_turns=["Round 1 critique", "Round 2 critique with details"],
        current_plan="# Plan",
    )
    assert "Round 2 critique" in summary


def test_issue_similarity_lexical_fallback(tmp_path):
    del tmp_path
    service = IssueSimilarityService(
        IssueDedupeConfig(
            embeddings_enabled="false",
            embedding_model="unused",
            similarity_threshold=0.9,
            fallback_mode="lexical",
        )
    )
    duplicate = service.find_duplicate(
        "Architecture violates layering boundaries",
        [("issue_1", "Architecture violates layering rules")],
    )
    assert duplicate == "issue_1"


def test_issue_similarity_embeddings_primary_path(tmp_path, monkeypatch):
    del tmp_path

    class FakeLiteLLM:
        @staticmethod
        def embedding(model, input):  # type: ignore[no-untyped-def]
            del model
            text = input[0]
            if "candidate" in text:
                vector = [1.0, 0.0, 0.0]
            else:
                vector = [0.99, 0.01, 0.0]
            return type("Resp", (), {"data": [{"embedding": vector}]})

    monkeypatch.setitem(__import__("sys").modules, "litellm", FakeLiteLLM)
    service = IssueSimilarityService(
        IssueDedupeConfig(
            embeddings_enabled="true",
            embedding_model="text-embedding-004",
            similarity_threshold=0.95,
            fallback_mode="none",
        )
    )
    duplicate = service.find_duplicate("candidate issue", [("issue_1", "existing issue")])
    assert duplicate == "issue_1"


@pytest.mark.asyncio
async def test_convergence_uses_stability_signals(tmp_path):
    runtime = _runtime(tmp_path)
    session = runtime.store.create_planning_session(
        repo_name="repo",
        title="convergence",
        requirements="r",
        seed_type="requirements",
        status="refining",
    )
    state = runtime._state(session.id, session)  # noqa: SLF001
    state.architecture_change_rounds = [True, False]
    state.review_score_history = [8.8, 8.9]
    state.open_issue_history = [3, 1]
    issue_tracker = state.issue_tracker
    assert issue_tracker is not None

    review = ReviewResult(
        strengths=[],
        architectural_concerns=[],
        risks=[],
        simplification_opportunities=[],
        blocking_issues=[],
        reviewer_questions=[],
        recommended_changes=[],
        design_quality_score=8.9,
        confidence="high",
        review_complete=True,
        simplest_possible_design=None,
        primary_issue=None,
        resolved_issues=[],
        constraint_violations=[],
        issue_priority=[],
        prose="",
    )

    async def emit_tool(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return None

    ctx = PlanningRoundContext(
        core=runtime._core(session.id),  # noqa: SLF001
        session_id=session.id,
        round_number=3,
        requirements="r",
        state=state,
        issue_tracker=issue_tracker,
        selected_author_model="gpt-4o-mini",
        selected_critic_model="gpt-4o-mini",
        event_callback=None,
    )
    implementability, convergence = await runtime._stage_convergence_check(  # noqa: SLF001
        ctx=ctx,
        updated_markdown="# plan",
        validation_review=review,
        emit_tool=emit_tool,
    )
    assert implementability.implementable is True
    assert convergence.converged is False
    assert convergence.reason == "stability_not_met"


@pytest.mark.asyncio
async def test_convergence_allows_stable_clean_round_without_explicit_review_complete(tmp_path):
    runtime = _runtime(tmp_path)
    session = runtime.store.create_planning_session(
        repo_name="repo",
        title="stable-clean",
        requirements="r",
        seed_type="requirements",
        status="refining",
    )
    state = runtime._state(session.id, session)  # noqa: SLF001
    state.architecture_change_rounds = [False, False]
    state.review_score_history = [7.9, 8.0]
    state.open_issue_history = [1, 0]
    issue_tracker = state.issue_tracker
    assert issue_tracker is not None

    review = ReviewResult(
        strengths=[],
        architectural_concerns=[],
        risks=[],
        simplification_opportunities=[],
        blocking_issues=[],
        reviewer_questions=[],
        recommended_changes=[],
        design_quality_score=8.0,
        confidence="high",
        review_complete=False,
        simplest_possible_design=None,
        primary_issue=None,
        resolved_issues=[],
        constraint_violations=[],
        issue_priority=[],
        prose="",
    )

    async def emit_tool(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return None

    async def fake_run_design_review(**kwargs):  # type: ignore[no-untyped-def]
        if kwargs.get("mode") == "implementability":
            return ImplementabilityResult(
                implementable=True,
                missing_details=[],
                implementation_risks=[],
                suggested_additions=[],
                prose="ok",
            )
        raise AssertionError("unexpected critic mode in convergence test")

    runtime.critic.run_design_review = fake_run_design_review  # type: ignore[method-assign]

    ctx = PlanningRoundContext(
        core=runtime._core(session.id),  # noqa: SLF001
        session_id=session.id,
        round_number=2,
        requirements="r",
        state=state,
        issue_tracker=issue_tracker,
        selected_author_model="gpt-4o-mini",
        selected_critic_model="gpt-4o-mini",
        event_callback=None,
    )
    implementability, convergence = await runtime._stage_convergence_check(  # noqa: SLF001
        ctx=ctx,
        updated_markdown="# plan",
        validation_review=review,
        emit_tool=emit_tool,
    )

    assert implementability.implementable is True
    assert convergence.converged is True
    assert convergence.reason == "review_complete"


@pytest.mark.asyncio
async def test_convergence_stops_stalled_refinement_loop(tmp_path):
    runtime = _runtime(tmp_path)
    session = runtime.store.create_planning_session(
        repo_name="repo",
        title="stalled",
        requirements="r",
        seed_type="requirements",
        status="refining",
    )
    state = runtime._state(session.id, session)  # noqa: SLF001
    state.architecture_change_rounds = [True, True]
    state.review_score_history = [7.0, 7.0]
    state.open_issue_history = [2, 2]
    issue_tracker = state.issue_tracker
    assert issue_tracker is not None
    issue_tracker.add_issue("Logging framework detail remains open", 1)
    issue_tracker.add_issue("Metrics naming detail remains open", 1)
    state.open_issue_history = [2, 2]

    review = ReviewResult(
        strengths=[],
        architectural_concerns=[],
        risks=[],
        simplification_opportunities=[],
        blocking_issues=[],
        reviewer_questions=[],
        recommended_changes=[],
        design_quality_score=7.0,
        confidence="medium",
        review_complete=False,
        simplest_possible_design=None,
        primary_issue="Logging framework detail remains open",
        resolved_issues=[],
        constraint_violations=[],
        issue_priority=[],
        prose="",
    )

    async def emit_tool(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return None

    async def fake_run_design_review(**kwargs):  # type: ignore[no-untyped-def]
        if kwargs.get("mode") == "implementability":
            return ImplementabilityResult(
                implementable=True,
                missing_details=[],
                implementation_risks=[],
                suggested_additions=[],
                prose="ok",
            )
        raise AssertionError("unexpected critic mode in stalled convergence test")

    runtime.critic.run_design_review = fake_run_design_review  # type: ignore[method-assign]

    ctx = PlanningRoundContext(
        core=runtime._core(session.id),  # noqa: SLF001
        session_id=session.id,
        round_number=3,
        requirements="r",
        state=state,
        issue_tracker=issue_tracker,
        selected_author_model="gpt-4o-mini",
        selected_critic_model="gpt-4o-mini",
        event_callback=None,
    )
    implementability, convergence = await runtime._stage_convergence_check(  # noqa: SLF001
        ctx=ctx,
        updated_markdown="# plan",
        validation_review=review,
        emit_tool=emit_tool,
    )

    assert implementability.implementable is True
    assert convergence.converged is True
    assert convergence.reason == "stalled_refinement"


def test_config_parses_issue_dedupe_settings(tmp_path):
    config_path = tmp_path / "prscope.yml"
    config_path.write_text(
        """
local_repo: .
planning:
  issue_dedupe:
    embeddings_enabled: true
    embedding_model: text-embedding-004
    similarity_threshold: 0.9
    fallback_mode: lexical
  issue_graph:
    max_nodes: 42
    max_edges: 84
    causality_extraction_enabled: true
    causality_max_edges_per_review: 5
    causality_min_text_len: 15
        """.strip()
    )
    config = PrscopeConfig.load(tmp_path)
    assert config.planning.issue_dedupe.embeddings_enabled == "true"
    assert config.planning.issue_dedupe.embedding_model == "text-embedding-004"
    assert config.planning.issue_dedupe.similarity_threshold == 0.9
    assert config.planning.issue_dedupe.fallback_mode == "lexical"
    assert config.planning.issue_graph.max_nodes == 42
    assert config.planning.issue_graph.max_edges == 84
    assert config.planning.issue_graph.causality_extraction_enabled is True
    assert config.planning.issue_graph.causality_max_edges_per_review == 5
    assert config.planning.issue_graph.causality_min_text_len == 15


def test_state_snapshot_persists_issue_graph_payload(tmp_path):
    runtime = _runtime(tmp_path)
    session = runtime.store.create_planning_session(
        repo_name="repo",
        title="snapshot",
        requirements="r",
        seed_type="requirements",
        status="draft",
    )
    state = runtime._state(session.id, session)  # noqa: SLF001
    tracker = state.issue_tracker
    assert tracker is not None
    root = tracker.add_issue("Architecture layering violation", 1)
    child = tracker.add_issue("Module dependency cycle", 1)
    tracker.add_edge(root.id, child.id, "causes")
    tracker.alias_duplicate("issue_alias_1", root.id)
    runtime._persist_state_snapshot(session.id)  # noqa: SLF001

    payload = runtime.read_state_snapshot(session.id)
    assert payload is not None
    issue_graph = payload.get("issue_graph")
    assert payload.get("schema_version") == 1
    assert isinstance(issue_graph, dict)
    assert "nodes" in issue_graph
    assert "edges" in issue_graph
    assert "duplicate_alias" in issue_graph


def test_state_snapshot_persists_round_histories(tmp_path):
    runtime = _runtime(tmp_path)
    session = runtime.store.create_planning_session(
        repo_name="repo",
        title="history-snapshot",
        requirements="r",
        seed_type="requirements",
        status="refining",
    )
    state = runtime._state(session.id, session)  # noqa: SLF001
    state.architecture_change_rounds = [True, False, True]
    state.review_score_history = [6.0, 7.0, 7.0]
    state.open_issue_history = [4, 2, 2]
    state.accessed_paths = {"prscope/web/api.py"}
    state.plan_markdown = "# plan"

    runtime._persist_state_snapshot(session.id)  # noqa: SLF001
    runtime.cleanup_session_resources(session.id)

    restored = runtime._state(session.id, session)  # noqa: SLF001
    assert restored.architecture_change_rounds == [True, False, True]
    assert restored.review_score_history == [6.0, 7.0, 7.0]
    assert restored.open_issue_history == [4, 2, 2]
    assert restored.accessed_paths == {"prscope/web/api.py"}
    assert restored.plan_markdown == "# plan"


def test_state_bootstrap_legacy_snapshot_dedupes_and_aliases(tmp_path):
    runtime = _runtime(tmp_path)
    session = runtime.store.create_planning_session(
        repo_name="repo",
        title="legacy",
        requirements="r",
        seed_type="requirements",
        status="draft",
    )
    snapshot_dir = tmp_path / ".prscope" / "sessions"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    legacy_payload = {
        "session_id": session.id,
        "updated_at": "2026-03-05T12:00:00Z",
        "open_issues": [
            {
                "id": "issue_2",
                "description": "Architecture violates layering boundaries",
                "status": "open",
                "raised_in_round": 1,
                "resolved_in_round": None,
            },
            {
                "id": "issue_5",
                "description": "Architecture violates layering boundaries",
                "status": "open",
                "raised_in_round": 1,
                "resolved_in_round": None,
            },
        ],
    }
    (snapshot_dir / f"{session.id}.json").write_text(json.dumps(legacy_payload), encoding="utf-8")

    state = runtime._state(session.id, session)  # noqa: SLF001
    tracker = state.issue_tracker
    assert tracker is not None
    graph = tracker.graph_snapshot()
    assert len(graph["nodes"]) == 1
    assert graph["duplicate_alias"].get("issue_5") == graph["nodes"][0]["id"]


def test_persist_snapshot_keeps_default_graph_on_malformed_tracker_payload(tmp_path):
    runtime = _runtime(tmp_path)
    session = runtime.store.create_planning_session(
        repo_name="repo",
        title="malformed-graph",
        requirements="r",
        seed_type="requirements",
        status="draft",
    )
    state = runtime._state(session.id, session)  # noqa: SLF001
    tracker = state.issue_tracker
    assert tracker is not None

    def _malformed_snapshot():  # type: ignore[no-untyped-def]
        return None

    tracker.graph_snapshot = _malformed_snapshot  # type: ignore[method-assign]
    runtime._persist_state_snapshot(session.id)  # noqa: SLF001
    payload = runtime.read_state_snapshot(session.id)
    assert payload is not None
    assert payload.get("schema_version") == 1
    graph = payload.get("issue_graph")
    assert isinstance(graph, dict)
    assert graph.get("nodes") == []
    assert graph.get("edges") == []
    assert graph.get("duplicate_alias") == {}


def test_state_cache_is_bounded_and_lru_like(tmp_path):
    runtime = _runtime(tmp_path)
    session_ids: list[str] = []
    for idx in range(205):
        session = runtime.store.create_planning_session(
            repo_name="repo",
            title=f"s-{idx}",
            requirements="r",
            seed_type="requirements",
            status="draft",
        )
        session_ids.append(session.id)
        runtime._state(session.id, session)  # noqa: SLF001
    assert len(runtime._states) <= 200  # noqa: SLF001

    hottest = session_ids[-1]
    assert hottest in runtime._states  # noqa: SLF001


def test_status_includes_issue_graph_health_summary(tmp_path):
    runtime = _runtime(tmp_path)
    session = runtime.store.create_planning_session(
        repo_name="repo",
        title="status",
        requirements="r",
        seed_type="requirements",
        status="refining",
    )
    core = runtime._core(session.id)  # noqa: SLF001
    core.save_plan_version(
        "# Plan\n\n## Files Changed\n- `a.py`\n- `b.py`\n",
        round_number=1,
    )
    state = runtime._state(session.id, session)  # noqa: SLF001
    tracker = state.issue_tracker
    assert tracker is not None
    root = tracker.add_issue("Architecture violation", 1, preferred_id="issue_1")
    child = tracker.add_issue("Dependency cycle", 1, preferred_id="issue_2")
    dep = tracker.add_issue("Test isolation blocked", 1, preferred_id="issue_3")
    tracker.add_edge(root.id, child.id, "causes")
    tracker.add_edge(child.id, dep.id, "depends_on")

    payload = runtime.status(session.id, {"a.py"})
    assert payload["open_issues"] == 3
    assert payload["root_open_issues"] == 2
    assert payload["dependency_blocks"] == 1
