from __future__ import annotations

import pytest

from prscope.config import IssueDedupeConfig, IssueGraphConfig, PlanningConfig, PrscopeConfig, RepoProfile
from prscope.planning.runtime.authoring.models import PlanDocument, render_markdown
from prscope.planning.runtime.critic import ReviewResult
from prscope.planning.runtime.orchestration import PlanningRuntime
from prscope.planning.runtime.pipeline.round_context import PlanningRoundContext
from prscope.planning.runtime.pipeline.stages import review_issue_severity
from prscope.planning.runtime.review import IssueCausalityExtractor, IssueGraphTracker, IssueSimilarityService
from prscope.store import Store


def _tracker() -> IssueGraphTracker:
    similarity = IssueSimilarityService(
        IssueDedupeConfig(
            embeddings_enabled="false",
            embedding_model="unused",
            similarity_threshold=0.95,
            fallback_mode="none",
        )
    )
    return IssueGraphTracker(similarity=similarity, max_nodes=50, max_edges=100)


def _review(*, prose: str) -> ReviewResult:
    return ReviewResult(
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
        primary_issue=None,
        resolved_issues=[],
        constraint_violations=[],
        issue_priority=[],
        prose=prose,
    )


def test_canonicalization_applies_to_add_edge_and_resolve():
    tracker = _tracker()
    root = tracker.add_issue("Architecture layering violation", 1, preferred_id="issue_1")
    child = tracker.add_issue("Module dependency cycle", 1, preferred_id="issue_2")
    tracker.alias_duplicate("issue_alias_1", root.id)
    tracker.add_edge("issue_alias_1", child.id, "causes")

    snapshot = tracker.graph_snapshot()
    assert snapshot["edges"] == [{"source": "issue_1", "target": "issue_2", "relation": "causes"}]

    tracker.resolve_issue("issue_alias_1", 2)
    open_ids = [issue.id for issue in tracker.open_issues()]
    assert "issue_1" not in open_ids
    assert "issue_2" not in open_ids


def test_resolution_propagation_uses_causes_and_dependency_gate():
    tracker = _tracker()
    a = tracker.add_issue("Architecture issue", 1, preferred_id="issue_1")
    b = tracker.add_issue("Layering issue", 1, preferred_id="issue_2")
    c = tracker.add_issue("Testing blocked", 1, preferred_id="issue_3")
    d = tracker.add_issue("Dependency fix required", 1, preferred_id="issue_4")

    tracker.add_edge(a.id, c.id, "causes")
    tracker.add_edge(b.id, c.id, "causes")
    tracker.add_edge(c.id, d.id, "depends_on")

    tracker.resolve_issue(a.id, 2)
    assert any(issue.id == c.id for issue in tracker.open_issues())

    tracker.resolve_issue(b.id, 2)
    assert any(issue.id == c.id for issue in tracker.open_issues())

    tracker.resolve_issue(d.id, 3)
    tracker.resolve_issue(b.id, 3)
    assert all(issue.id != c.id for issue in tracker.open_issues())


def test_shallow_resolution_does_not_propagate_to_causal_children():
    tracker = _tracker()
    root = tracker.add_issue("Root issue", 1, preferred_id="issue_1")
    child = tracker.add_issue("Child issue", 1, preferred_id="issue_2")
    tracker.add_edge(root.id, child.id, "causes")

    tracker.resolve_issue(root.id, 2, propagate_causes=False, resolution_source="lightweight")

    snapshot = tracker.graph_snapshot()
    statuses = {node["id"]: node["status"] for node in snapshot["nodes"]}
    resolution_sources = {node["id"]: node.get("resolution_source") for node in snapshot["nodes"]}
    assert statuses["issue_1"] == "resolved"
    assert statuses["issue_2"] == "open"
    assert resolution_sources["issue_1"] == "lightweight"
    assert resolution_sources["issue_2"] is None


def test_propagating_resolution_marks_review_flow_source():
    tracker = _tracker()
    root = tracker.add_issue("Root issue", 1, preferred_id="issue_1")
    child = tracker.add_issue("Child issue", 1, preferred_id="issue_2")
    tracker.add_edge(root.id, child.id, "causes")

    tracker.resolve_issue(root.id, 2)

    resolution_sources = {node["id"]: node.get("resolution_source") for node in tracker.graph_snapshot()["nodes"]}
    assert resolution_sources["issue_1"] == "review"
    assert resolution_sources["issue_2"] == "review"


def test_root_open_excludes_dependency_edges():
    tracker = _tracker()
    a = tracker.add_issue("Architecture issue", 1, preferred_id="issue_1")
    b = tracker.add_issue("Testing issue", 1, preferred_id="issue_2")
    tracker.add_edge(b.id, a.id, "depends_on")

    roots = {issue.id for issue in tracker.root_open_issues()}
    assert roots == {"issue_1", "issue_2"}


def test_unresolved_dependency_chain_count():
    tracker = _tracker()
    a = tracker.add_issue("Root issue", 1, preferred_id="issue_1")
    b = tracker.add_issue("Derived issue", 1, preferred_id="issue_2")
    tracker.add_edge(b.id, a.id, "depends_on")
    assert tracker.unresolved_dependency_chains() == 1
    tracker.resolve_issue(a.id, 2)
    assert tracker.unresolved_dependency_chains() == 0


def test_graph_snapshot_includes_duplicate_alias():
    tracker = _tracker()
    root = tracker.add_issue("Root issue", 1, preferred_id="issue_1")
    tracker.alias_duplicate("issue_alias_9", root.id)
    snapshot = tracker.graph_snapshot()
    assert snapshot["duplicate_alias"] == {"issue_alias_9": "issue_1"}


def test_causality_extractor_adds_causal_edges_with_guardrails():
    tracker = _tracker()
    graph_config = IssueGraphConfig(
        causality_extraction_enabled=True,
        causality_max_edges_per_review=4,
        causality_min_text_len=12,
    )
    extractor = IssueCausalityExtractor(graph_config)
    review = _review(prose=("Testing fails because architecture violates layering boundaries. This leads to issues."))
    result = extractor.extract_edges(graph=tracker, review=review, round_number=1)
    snapshot = tracker.graph_snapshot()
    assert result.accepted_edges == 1
    assert len(snapshot["edges"]) == 1
    assert snapshot["edges"][0]["relation"] == "causes"
    node_severities = {node["severity"] for node in snapshot["nodes"]}
    assert node_severities == {"minor"}


def test_review_issue_severity_demotes_architectural_and_implementability_items():
    assert review_issue_severity("blocking_issue") == "major"
    assert review_issue_severity("constraint_violation") == "major"
    assert review_issue_severity("architectural_concern") == "minor"
    assert review_issue_severity("validation_architectural_concern") == "minor"
    assert review_issue_severity("implementability_detail") == "minor"


def test_issue_graph_snapshot_persists_related_decision_ids():
    tracker = _tracker()
    issue = tracker.add_issue(
        "Database choice remains underspecified.",
        1,
        preferred_id="issue_1",
        issue_type="ambiguity",
    )

    tracker.link_issue_to_decisions(issue.id, ["architecture.database"], relation="missing")

    snapshot = tracker.graph_snapshot()
    assert snapshot["nodes"][0]["related_decision_ids"] == ["architecture.database"]
    assert snapshot["nodes"][0]["issue_type"] == "ambiguity"
    assert "decision:missing" in snapshot["nodes"][0]["tags"]

    restored = _tracker()
    restored.load_snapshot(snapshot)
    restored_snapshot = restored.graph_snapshot()
    assert restored_snapshot["nodes"][0]["related_decision_ids"] == ["architecture.database"]
    assert restored_snapshot["nodes"][0]["issue_type"] == "ambiguity"
    assert "decision:missing" in restored_snapshot["nodes"][0]["tags"]


@pytest.mark.asyncio
async def test_design_review_links_issues_to_related_decisions(tmp_path):
    store = Store(tmp_path / "test.db")
    config = PrscopeConfig(
        local_repo=str(tmp_path),
        planning=PlanningConfig(author_model="gpt-4o-mini", critic_model="gpt-4o-mini"),
    )
    repo = RepoProfile(name="repo", path=str(tmp_path))
    runtime = PlanningRuntime(store=store, config=config, repo=repo)

    session = store.create_planning_session(
        repo_name=repo.name,
        title="decision-links",
        requirements="r",
        seed_type="chat",
        status="refining",
    )
    plan = PlanDocument(
        title="Plan",
        summary="Add persistence.",
        goals="Support durable state.",
        non_goals="Do not change the external API.",
        files_changed="`src/app.py` - wire persistence.",
        architecture="Keep the storage layer abstract until the database decision is confirmed.",
        implementation_steps="1. Add the storage abstraction.",
        test_strategy="Add integration tests for persistence.",
        rollback_plan="Revert the storage integration if rollout fails.",
        open_questions="- Which database should store the primary application data?",
    )
    version = runtime._core(session.id).save_plan_version(
        render_markdown(plan),
        round_number=1,
        plan_document=plan,
    )
    runtime._attach_plan_version_artifacts(
        version_id=version.id,
        plan_document=plan,
        plan_content=version.plan_content,
    )
    state = runtime._state(session.id)
    ctx = PlanningRoundContext(
        core=runtime._core(session.id),
        session_id=session.id,
        round_number=1,
        requirements="r",
        state=state,
        issue_tracker=state.issue_tracker,
        selected_author_model="gpt-4o-mini",
        selected_critic_model="gpt-4o-mini",
        event_callback=None,
    )

    async def fake_run_design_review(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        return ReviewResult(
            strengths=[],
            architectural_concerns=["Database choice remains underspecified for persistence rollout."],
            risks=[],
            simplification_opportunities=[],
            blocking_issues=[],
            reviewer_questions=[],
            recommended_changes=[],
            design_quality_score=6.0,
            confidence="medium",
            review_complete=False,
            simplest_possible_design=None,
            primary_issue="Database choice remains underspecified for persistence rollout.",
            resolved_issues=[],
            constraint_violations=[],
            issue_priority=[],
            prose="Database choice remains underspecified for persistence rollout.",
            parse_error=None,
        )

    runtime.critic.run_design_review = fake_run_design_review  # type: ignore[method-assign]

    async def emit_tool(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs

    await runtime._stages.design_review(  # noqa: SLF001
        ctx=ctx,
        current_plan_content=version.plan_content,
        emit_tool=emit_tool,
    )

    linked_nodes = {
        node["id"]: node for node in ctx.issue_tracker.graph_snapshot()["nodes"] if node.get("related_decision_ids")
    }
    assert linked_nodes
    assert any(node["related_decision_ids"] == ["architecture.database"] for node in linked_nodes.values())
    assert any("decision:missing" in node.get("tags", []) for node in linked_nodes.values())
