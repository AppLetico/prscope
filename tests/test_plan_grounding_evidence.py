"""Tests for plan grounding: server vs localized UI, Files Changed allowlist."""

from __future__ import annotations

from prscope.planning.runtime.authoring.discovery import (
    PRSCOPE_HTTP_CLIENT_HINTS,
    is_server_api_or_auth_request,
    requirements_imply_http_client_call_sites,
)
from prscope.planning.runtime.authoring.models import EvidenceBundle
from prscope.planning.runtime.authoring.pipeline import AuthorPlannerPipeline
from prscope.planning.runtime.authoring.planner_paths import planner_verified_file_paths
from prscope.planning.runtime.authoring.validation import AuthorValidationService
from prscope.planning.runtime.pipeline.stages import PlanningStages


def test_is_server_api_or_auth_true_for_boundary_not_fastapi_alone() -> None:
    assert is_server_api_or_auth_request("Add middleware to protect /api/ routes with PRSCOPE_API_KEY")
    assert is_server_api_or_auth_request("Enforce bearer tokens on FastAPI /api/ endpoints")
    assert not is_server_api_or_auth_request(
        "FastAPI localized change: adjust session snapshot payload for PlanningView and PlanPanel."
    )


def test_requirements_imply_http_client_call_sites() -> None:
    assert requirements_imply_http_client_call_sites("Update first-party callers and e2e to send the key")
    assert not requirements_imply_http_client_call_sites("Add a health check endpoint")


def test_files_changed_subset_failures() -> None:
    md = "## Files Changed\n- `src/oops/wrong.py`\n- `src/ok/right.py`\n"
    allow = {"src/ok/right.py", "src/other.py"}
    failures = AuthorValidationService.files_changed_subset_failures(md, allow)
    assert failures
    assert "outside evidence allowlist" in failures[0]


def test_merge_refinement_allowlist_with_verified_paths_unions_session_anchors() -> None:
    """Session-verified paths (prior plan / tool reads) must widen the repo allowlist."""
    base = {"src/a.py", "src/b.py"}
    verified = {"src/c.py", "src/a.py"}
    merged = PlanningStages._merge_refinement_allowlist_with_verified_paths(base, verified)
    assert merged == {"src/a.py", "src/b.py", "src/c.py"}


def test_merge_refinement_allowlist_with_verified_paths_preserves_none() -> None:
    assert PlanningStages._merge_refinement_allowlist_with_verified_paths(None, {"x.py"}) is None


def test_requirement_named_paths_count_as_verified_for_unknown_refs() -> None:
    """Paths the user names in plain text must not fail grounding/unknown-ref checks."""
    from types import SimpleNamespace

    req = "Extend tests/test_web_api_models.py for the new behavior; touch AGENTS.md."
    svc = AuthorValidationService(SimpleNamespace(read_history={}))
    ru = SimpleNamespace(
        file_contents={},
        entrypoints=[],
        core_modules=[],
        relevant_modules=[],
        relevant_tests=[],
    )
    plan = (
        "# T\n## Goals\n- g\n## Non-Goals\n- n\n## Files Changed\n"
        "- `tests/test_web_api_models.py`\n"
        "## Architecture\n- a\n"
        "## Example Code Snippets\n```python\npass\n```\n"
    )
    result = svc.validate_draft_result(
        plan_content=plan,
        repo_understanding=ru,
        draft_phase="planner",
        min_grounding_ratio=0.35,
        requirements_text=req,
        planner_complexity="simple",
    )
    assert not any("unknown file references" in f for f in result.failure_messages)


def test_paths_mentioned_in_requirements_plain_filenames() -> None:
    req = "Update AGENTS.md and docs/agent-harness.md only; no code."
    paths = AuthorValidationService.paths_mentioned_in_requirements(req)
    assert "AGENTS.md" in paths
    assert "docs/agent-harness.md" in paths


def test_build_files_changed_evidence_allowlist_includes_requirement_named_paths() -> None:
    req = "Update AGENTS.md and docs/agent-harness.md only; no code."
    allow = AuthorValidationService.build_files_changed_evidence_allowlist(
        relevant_files=(),
        test_targets=(),
        related_modules=(),
        http_client_hints=(),
        grounding_paths=set(),
        requirements_text=req,
    )
    assert allow is not None
    assert "AGENTS.md" in allow
    assert "docs/agent-harness.md" in allow


def test_build_files_changed_evidence_allowlist_includes_http_hints() -> None:
    allow = AuthorValidationService.build_files_changed_evidence_allowlist(
        relevant_files=("src/a.py",),
        test_targets=("tests/t.py",),
        related_modules=(),
        http_client_hints=PRSCOPE_HTTP_CLIENT_HINTS,
        grounding_paths=set(),
    )
    assert allow is not None
    assert "src/prscope/benchmark.py" in allow


def test_planner_verified_file_paths_dedupes_and_limits() -> None:
    from types import SimpleNamespace

    ru = SimpleNamespace(
        relevant_modules=[f"m{i}.py" for i in range(45)],
        relevant_tests=(),
        file_contents={},
        entrypoints=(),
        core_modules=(),
    )
    assert len(planner_verified_file_paths(ru, limit=40)) == 40


def test_planner_files_changed_allowlist_union_includes_verified_paths() -> None:
    """Paths in Verified File Paths (file_contents) must pass Files Changed subset check when merged."""
    from types import SimpleNamespace

    ru = SimpleNamespace(
        relevant_modules=("src/a.py",),
        relevant_tests=(),
        file_contents={"src/prscope/config.py": "x"},
        entrypoints=(),
        core_modules=(),
    )
    base = AuthorValidationService.build_files_changed_evidence_allowlist(
        relevant_files=("src/a.py",),
        test_targets=("tests/t.py",),
        related_modules=(),
        http_client_hints=(),
        grounding_paths=set(),
    )
    assert base is not None
    assert "src/prscope/config.py" not in base
    merged = base | set(planner_verified_file_paths(ru, limit=40))
    assert "src/prscope/config.py" in merged
    md = "## Files Changed\n- `src/prscope/config.py`\n"
    assert not AuthorValidationService.files_changed_subset_failures(md, merged)


def test_localized_frontend_owner_paths_empty_for_api_key_prompt() -> None:
    eb = EvidenceBundle(
        relevant_files=(
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/components/ActionBar.tsx",
        ),
        existing_components=(),
        test_targets=(),
        related_modules=(),
        existing_routes_or_helpers=(),
        evidence_notes=(),
    )
    req = "Add API key auth for /api/*; update first-party UI fetch and EventSource"
    assert AuthorPlannerPipeline._localized_frontend_owner_paths(eb, req) == ()
