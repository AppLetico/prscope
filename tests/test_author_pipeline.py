from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prscope.config import PlanningConfig
from prscope.planning.runtime.author import AuthorAgent, RepoUnderstanding
from prscope.planning.runtime.authoring.discovery import requirement_search_patterns
from prscope.planning.runtime.authoring.models import (
    AttemptContext,
    AuthorResult,
    EvidenceBundle,
    PlanDocument,
    RepairPlan,
    RepoCandidates,
    ValidationResult,
)
from prscope.planning.runtime.authoring import pipeline as pipeline_module
from prscope.planning.runtime.authoring.pipeline import AuthorPlannerPipeline
from prscope.planning.runtime.authoring.repair import AuthorRepairService
from prscope.planning.runtime.tools import ToolExecutor


def _make_agent(repo_root: Path) -> AuthorAgent:
    executor = ToolExecutor(repo_root)
    return AuthorAgent(config=PlanningConfig(author_model="gpt-4o-mini"), tool_executor=executor)


def test_scan_repo_candidates_discovers_entrypoints_and_configs(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('hi')\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "service.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_service.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[tool.pytest]\n", encoding="utf-8")

    agent = _make_agent(tmp_path)
    candidates = agent.scan_repo_candidates(mental_model="See `src/service.py`")

    assert "app.py" in candidates.entrypoints
    assert "src/service.py" in candidates.source_modules
    assert "pyproject.toml" in candidates.tests_and_config
    assert "tests/test_service.py" in candidates.tests_and_config
    assert "tests/test_service.py" not in candidates.source_modules
    assert "src/service.py" in candidates.all_paths


def test_explore_repo_reads_ranked_files(tmp_path: Path) -> None:
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "server.py").write_text("def start_server():\n    return 'ok'\n", encoding="utf-8")
    (tmp_path / "api" / "handlers.py").write_text("def handle_request():\n    return 200\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[tool]\n", encoding="utf-8")

    agent = _make_agent(tmp_path)
    candidates = agent.scan_repo_candidates(mental_model="")
    understanding = agent.explore_repo(
        requirements="add api request handling and server startup",
        candidates=candidates,
    )

    assert understanding.file_contents
    assert any(path.endswith(".py") for path in understanding.file_contents)
    assert understanding.architecture_summary
    assert isinstance(understanding.risks, list)


def test_requirement_search_patterns_prioritize_route_literals_and_keywords() -> None:
    patterns = requirement_search_patterns("Add a lightweight /health endpoint and tests for it.")

    assert r"/health" in patterns
    assert r"\bhealth\b" in patterns
    assert all("endpoint" not in pattern for pattern in patterns)


def test_explore_repo_reads_around_grep_hits_for_deep_route_definitions(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "api.py").write_text(
        "\n".join(
            [
                "from fastapi import FastAPI",
                "app = FastAPI()",
                *[f'# filler {idx}' for idx in range(1, 190)],
                '@app.get(\"/health\")',
                "async def health():",
                "    return {\"status\": \"healthy\"}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_api.py").write_text(
        "def test_health():\n    assert True\n",
        encoding="utf-8",
    )

    agent = _make_agent(tmp_path)
    candidates = agent.scan_repo_candidates(mental_model="")
    understanding = agent.explore_repo(
        requirements="Add a lightweight /health endpoint and tests for it.",
        candidates=candidates,
    )

    api_contents = understanding.file_contents.get("src/api.py", "")
    assert "/health" in api_contents
    assert "async def health" in api_contents


def test_explore_repo_prefers_source_files_and_related_tests_for_endpoint_work(tmp_path: Path) -> None:
    (tmp_path / "src" / "prscope" / "web").mkdir(parents=True)
    (tmp_path / "src" / "prscope" / "web" / "api.py").write_text(
        '@app.get("/health")\nasync def health():\n    return {"status": "healthy"}\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "prscope" / "web" / "server.py").write_text("from .api import create_app\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_web_api_models.py").write_text(
        'def test_health_endpoint_returns_healthy():\n    response = client.get("/health")\n',
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_discovery_parser.py").write_text(
        'payload = {"text": "Add a lightweight /health endpoint and tests for it."}\n',
        encoding="utf-8",
    )

    agent = _make_agent(tmp_path)
    candidates = agent.scan_repo_candidates(mental_model="")
    understanding = agent.explore_repo(
        requirements="Add a lightweight /health endpoint and tests for it.",
        candidates=candidates,
    )

    assert "src/prscope/web/api.py" in understanding.file_contents
    assert "tests/test_web_api_models.py" in understanding.file_contents


def test_explore_repo_prefers_frontend_page_and_api_helpers_for_localized_ui_work(tmp_path: Path) -> None:
    (tmp_path / "src" / "prscope" / "web" / "frontend" / "src" / "components").mkdir(parents=True)
    (tmp_path / "src" / "prscope" / "web" / "frontend" / "src" / "pages").mkdir(parents=True)
    (tmp_path / "src" / "prscope" / "web" / "frontend" / "src" / "lib").mkdir(parents=True)
    (tmp_path / "src" / "prscope" / "planning" / "runtime" / "authoring").mkdir(parents=True)
    (tmp_path / "src" / "prscope" / "web").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "prscope" / "web" / "frontend" / "src" / "components" / "ActionBar.tsx").write_text(
        "export function ActionBar() { return null; }\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "prscope" / "web" / "frontend" / "src" / "pages" / "PlanningView.tsx").write_text(
        "import { exportSession, downloadFile } from '../lib/api';\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "prscope" / "web" / "frontend" / "src" / "lib" / "api.ts").write_text(
        "export function exportSession() {}\nexport function downloadFile() {}\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "prscope" / "web" / "frontend" / "src" / "pages" / "SessionList.tsx").write_text(
        "export function SessionList() { return null; }\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "prscope" / "web" / "frontend" / "src" / "lib" / "markdown.ts").write_text(
        "export function renderMarkdown() {}\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "prscope" / "planning" / "runtime" / "authoring" / "discovery.py").write_text(
        "def unrelated_runtime_helper():\n    return 'planning'\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "prscope" / "web" / "api.py").write_text(
        "def download_export():\n    return 'ok'\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "prscope" / "web" / "frontend" / "src" / "pages" / "PlanningView.test.ts").write_text(
        "test('export', () => {})\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "prscope" / "web" / "frontend" / "src" / "components" / "ChatPanel.test.ts").write_text(
        "test('chat', () => {})\n",
        encoding="utf-8",
    )

    agent = _make_agent(tmp_path)
    candidates = agent.scan_repo_candidates(mental_model="")
    understanding = agent.explore_repo(
        requirements=(
            "Move export controls into the top ActionBar so users can download PRD and conversation markdown "
            "from anywhere on the planning page. Reuse the existing backend export/download endpoints and "
            "frontend export helpers instead of creating new endpoints."
        ),
        candidates=candidates,
    )

    assert "src/prscope/web/frontend/src/components/ActionBar.tsx" in understanding.file_contents
    assert "src/prscope/web/frontend/src/pages/PlanningView.tsx" in understanding.file_contents
    assert "src/prscope/web/frontend/src/lib/api.ts" in understanding.file_contents
    assert "src/prscope/web/frontend/src/pages/PlanningView.test.ts" in understanding.file_contents
    assert "src/prscope/web/frontend/src/pages/SessionList.tsx" not in understanding.file_contents
    assert "src/prscope/planning/runtime/authoring/discovery.py" not in understanding.file_contents
    assert "src/prscope/web/frontend/src/components/ChatPanel.test.ts" not in understanding.file_contents


def test_explore_repo_prefers_planning_view_health_paths_for_diagnostics_ui_work(tmp_path: Path) -> None:
    (tmp_path / "src" / "prscope" / "web" / "frontend" / "src" / "components").mkdir(parents=True)
    (tmp_path / "src" / "prscope" / "web" / "frontend" / "src" / "pages").mkdir(parents=True)
    (tmp_path / "src" / "prscope" / "web" / "frontend" / "src" / "lib").mkdir(parents=True)
    (tmp_path / "src" / "prscope" / "web").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "prscope").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "prscope" / "web" / "frontend" / "src" / "components" / "ActionBar.tsx").write_text(
        "export function ActionBar() { return null; }\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "prscope" / "web" / "frontend" / "src" / "components" / "PlanPanel.tsx").write_text(
        "export function PlanPanel() { return null; }\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "prscope" / "web" / "frontend" / "src" / "pages" / "PlanningView.tsx").write_text(
        "const snapshotQuery = getSessionSnapshot('1');\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "prscope" / "web" / "frontend" / "src" / "lib" / "api.ts").write_text(
        "export function getSessionSnapshot() {}\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "prscope" / "web" / "api.py").write_text(
        '@app.get("/api/sessions/{session_id}/diagnostics")\n'
        "async def get_session_diagnostics():\n    return {}\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "prscope" / "cli.py").write_text("def main():\n    return 0\n", encoding="utf-8")
    (tmp_path / "src" / "prscope" / "web" / "frontend" / "src" / "pages" / "PlanningView.test.ts").write_text(
        "test('diagnostics', () => {})\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "prscope" / "web" / "frontend" / "src" / "lib" / "decisionGraphRender.test.ts").write_text(
        "test('graph', () => {})\n",
        encoding="utf-8",
    )

    agent = _make_agent(tmp_path)
    candidates = agent.scan_repo_candidates(mental_model="")
    understanding = agent.explore_repo(
        requirements=(
            "Add a compact diagnostics dropdown to the top ActionBar that shows the snapshot updated time, "
            "open issue count, and constraint violation count, plus an action that opens the existing session "
            "diagnostics endpoint in a new tab. Reuse existing snapshot query data from PlanningView and the "
            "existing diagnostics/snapshot backend endpoints."
        ),
        candidates=candidates,
    )

    assert "src/prscope/web/frontend/src/components/ActionBar.tsx" in understanding.file_contents
    assert "src/prscope/web/frontend/src/pages/PlanningView.tsx" in understanding.file_contents
    assert "src/prscope/web/frontend/src/components/PlanPanel.tsx" in understanding.file_contents
    assert "src/prscope/web/frontend/src/lib/api.ts" in understanding.file_contents
    assert "src/prscope/web/frontend/src/pages/PlanningView.test.ts" in understanding.file_contents
    assert "src/prscope/cli.py" not in understanding.file_contents
    assert "src/prscope/web/frontend/src/lib/decisionGraphRender.test.ts" not in understanding.file_contents


def test_classify_complexity_uses_module_count_and_keywords(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    simple = RepoUnderstanding(
        entrypoints=["app.py"],
        core_modules=["app.py"],
        relevant_modules=["app.py"],
        relevant_tests=[],
        architecture_summary="",
        risks=[],
        file_contents={"app.py": "print('x')"},
        from_mental_model=False,
    )
    complex_repo = RepoUnderstanding(
        entrypoints=["app.py"],
        core_modules=[f"m{i}.py" for i in range(8)],
        relevant_modules=[f"m{i}.py" for i in range(8)],
        relevant_tests=[],
        architecture_summary="",
        risks=[],
        file_contents={f"m{i}.py": "x" for i in range(8)},
        from_mental_model=False,
    )

    assert agent.classify_complexity(requirements="fix typo", repo_understanding=simple) == "simple"
    assert (
        agent.classify_complexity(
            requirements="architect a new orchestration pipeline with scalability constraints",
            repo_understanding=complex_repo,
        )
        == "complex"
    )


def test_parse_plan_document_accepts_new_markdown_native_fields(tmp_path: Path) -> None:
    _ = tmp_path
    payload = {
        "title": "Plan",
        "summary": "Summary",
        "goals": "Goals",
        "non_goals": "Non-Goals",
        "files_changed": "- `a.py`",
        "architecture": "Architecture",
        "implementation_steps": "1. Step",
        "test_strategy": "Tests",
        "rollback_plan": "Rollback",
        "open_questions": "None",
    }
    raw = json.dumps(payload)
    parsed = AuthorAgent._parse_plan_document(raw)
    assert parsed.non_goals == "Non-Goals"
    assert "a.py" in parsed.files_changed


@pytest.mark.asyncio
async def test_revise_plan_recovers_from_trailing_commas_in_json() -> None:
    captured: dict[str, object] = {}

    async def _fake_llm_call(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        captured["messages"] = args[0] if args else None
        del kwargs
        raw = """{
  "problem_understanding": "Need to clarify logging behavior.",
  "updates": {
    "open_questions": "- None.",
  },
  "justification": {
    "open_questions": "Feedback answered the remaining ambiguity.",
  },
  "review_prediction": "Reviewer should accept.",
}"""
        return (
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw))]),
            "gpt-4o-mini",
        )

    service = AuthorRepairService(_fake_llm_call)
    current_plan = PlanDocument(
        title="Plan",
        summary="Summary",
        goals="Goals",
        non_goals="Non-goals",
        files_changed="- `a.py`",
        architecture="Architecture",
        implementation_steps="1. Step",
        test_strategy="Tests",
        rollback_plan="Rollback",
        open_questions="- TBD",
    )
    repair_plan = RepairPlan(
        problem_understanding="Need to resolve question.",
        accepted_issues=["open questions unresolved"],
        rejected_issues=[],
        root_causes=["missing decision"],
        repair_strategy="Update open questions section",
        target_sections=["open_questions"],
        revision_plan="Set to none",
    )

    result = await service.revise_plan(
        repair_plan=repair_plan,
        current_plan=current_plan,
        requirements="Keep behavior stable.",
        revision_budget=2,
        revision_hints=["Preserve `src/prscope/web/frontend/src/lib/api.ts` exactly as written."],
    )

    assert result.problem_understanding
    assert result.updates.get("open_questions") == "- None."
    assert "open_questions" in result.justification
    messages = captured["messages"]
    assert isinstance(messages, list)
    system_prompt = str(messages[0]["content"])
    user_prompt = str(messages[1]["content"])
    assert "Preserve exact grounded file paths, helper names, and endpoint names" in system_prompt
    assert "`src/prscope/web/frontend/src/lib/api.ts` and `exportSession` exactly as written" in system_prompt
    assert "localized UI or API-wiring requests that reuse existing helpers/endpoints" in system_prompt
    assert "observability/telemetry work" in system_prompt
    assert "dedicated hooks/contexts" in system_prompt
    assert "Preserve `src/prscope/web/frontend/src/lib/api.ts` exactly as written." in user_prompt


def test_incremental_grounding_failures_detects_new_unverified_paths(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)

    failures = agent.incremental_grounding_failures(
        previous_plan_content=(
            "# Plan\n\n## Files Changed\n- `src/prscope/web/frontend/src/lib/api.ts`\n"
        ),
        updated_plan_content=(
            "# Plan\n\n## Files Changed\n- `src/prscope/web/frontend/src/lib/api.ts`\n"
            "- `src/prscope/web/lib/api.ts`\n"
        ),
        verified_paths={"src/prscope/web/frontend/src/lib/api.ts"},
    )

    assert failures == [
        "revision introduced unverified file references: src/prscope/web/lib/api.ts",
    ]


def test_validate_draft_flags_observability_drift_for_localized_ui_requests(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    repo_understanding = RepoUnderstanding(
        entrypoints=[],
        core_modules=["src/prscope/web/frontend/src/components/ActionBar.tsx"],
        relevant_modules=[
            "src/prscope/web/frontend/src/components/ActionBar.tsx",
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
        ],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="frontend action bar integration",
        risks=[],
        file_contents={
            "src/prscope/web/frontend/src/components/ActionBar.tsx": "export function ActionBar() {}",
            "src/prscope/web/frontend/src/pages/PlanningView.tsx": "onExport()",
        },
        from_mental_model=False,
    )

    failures = agent.validate_draft(
        plan_content=(
            "# Plan\n\n## Summary\nMove export controls.\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n"
            "## Changes\n- z\n\n## Files Changed\n- `src/prscope/web/frontend/src/components/ActionBar.tsx`\n\n"
            "## Architecture\nObservability requirements will remain unaltered while existing logging continues.\n\n"
            "## Open Questions\n- None.\n"
        ),
        repo_understanding=repo_understanding,
        draft_phase="planner",
        requirements_text=(
            "Move export controls into the top ActionBar and reuse existing backend endpoints and frontend export helpers."
        ),
    )

    assert failures == [
        "localized UI/API draft introduced observability or telemetry scope not present in requirements; "
        "remove logging/telemetry/observability wording and keep the plan focused on existing UI wiring, "
        "API helper reuse, and tests"
    ]


def test_validate_draft_result_normalizes_reason_codes_and_retryability(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/api.py"],
        core_modules=["src/prscope/web/api.py"],
        relevant_modules=["src/prscope/web/api.py"],
        relevant_tests=["tests/test_web_api_models.py"],
        architecture_summary="single FastAPI module",
        risks=[],
        file_contents={"src/prscope/web/api.py": '@app.get("/health")\nasync def health():\n    return {"status": "ok"}\n'},
        from_mental_model=False,
    )

    result = agent.validate_draft_result(
        plan_content=(
            "# Draft\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Files Changed\n- `src/unknown.py`\n\n"
            "## Architecture\n- z\n\n## Implementation Steps\n1. change it\n"
        ),
        repo_understanding=repo_understanding,
        draft_phase="planner",
        min_grounding_ratio=0.5,
        requirements_text="Add a lightweight /health endpoint and tests for it.",
    )

    assert not result.ok
    assert result.retryable
    assert "missing_sections" in result.reason_codes
    assert "unknown_file_reference" in result.reason_codes
    assert result.normalized_signature == frozenset(result.reason_codes)
    assert result.failure_count >= 2


def test_validate_draft_result_requires_verified_test_target_when_tests_requested(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        core_modules=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        relevant_modules=[
            "src/prscope/web/frontend/src/components/ActionBar.tsx",
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/lib/api.ts",
        ],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="frontend action bar export flow",
        risks=[],
        file_contents={
            "src/prscope/web/frontend/src/pages/PlanningView.tsx": "export function PlanningViewPage() {}",
        },
        from_mental_model=False,
    )

    result = agent.validate_draft_result(
        plan_content=(
            "# Draft\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Files Changed\n"
            "- `src/prscope/web/frontend/src/components/ActionBar.tsx`\n"
            "- `src/prscope/web/frontend/src/pages/PlanningView.tsx`\n"
            "- `src/prscope/web/frontend/src/lib/api.ts`\n\n"
            "## Architecture\nReuse `exportSession` and `downloadFile` without adding new routes.\n\n"
            "## Open Questions\n- None.\n"
        ),
        repo_understanding=repo_understanding,
        draft_phase="planner",
        min_grounding_ratio=0.35,
        requirements_text=(
            "Move export controls into the top ActionBar and reuse existing backend export/download endpoints and frontend export helpers, and add tests."
        ),
    )

    assert not result.ok
    assert "missing_tests" in result.reason_codes
    assert any("PlanningView.test.ts" in failure for failure in result.failure_messages)


def test_validate_draft_result_suggests_closest_verified_path_for_unknown_reference(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        core_modules=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        relevant_modules=[
            "src/prscope/web/frontend/src/components/ActionBar.tsx",
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/lib/api.ts",
        ],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="frontend action bar export flow",
        risks=[],
        file_contents={
            "src/prscope/web/frontend/src/pages/PlanningView.tsx": "export function PlanningViewPage() {}",
        },
        from_mental_model=False,
    )

    result = agent.validate_draft_result(
        plan_content=(
            "# Draft\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Files Changed\n"
            "- `src/prscope/web/frontend/src/components/ActionBar.tsx`\n"
            "- `src/prscope/web/lib/api.ts`\n"
            "- `src/prscope/web/frontend/src/pages/PlanningView.test.ts`\n\n"
            "## Architecture\nReuse `exportSession` and `downloadFile` without adding new routes.\n\n"
            "## Open Questions\n- None.\n"
        ),
        repo_understanding=repo_understanding,
        draft_phase="planner",
        min_grounding_ratio=0.35,
        requirements_text=(
            "Move export controls into the top ActionBar and reuse existing backend export/download endpoints and frontend export helpers, and add tests."
        ),
    )

    assert not result.ok
    assert "unknown_file_reference" in result.reason_codes
    assert any(
        "replace unverified path `src/prscope/web/lib/api.ts` with `src/prscope/web/frontend/src/lib/api.ts`"
        in failure
        for failure in result.failure_messages
    )


def test_validate_draft_result_requires_explicit_snapshot_helper_reuse_for_diagnostics_prompt(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        core_modules=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        relevant_modules=[
            "src/prscope/web/frontend/src/components/ActionBar.tsx",
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/lib/api.ts",
        ],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="frontend diagnostics dropdown flow",
        risks=[],
        file_contents={
            "src/prscope/web/frontend/src/pages/PlanningView.tsx": "queryFn: () => getSessionSnapshot(id)",
            "src/prscope/web/frontend/src/lib/api.ts": "export function getSessionSnapshot() {}",
        },
        from_mental_model=False,
    )

    result = agent.validate_draft_result(
        plan_content=(
            "# Draft\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Files Changed\n"
            "- `src/prscope/web/frontend/src/components/ActionBar.tsx`\n"
            "- `src/prscope/web/frontend/src/lib/api.ts`\n"
            "- `src/prscope/web/frontend/src/pages/PlanningView.test.ts`\n\n"
            "## Architecture\nReuse the existing diagnostics endpoint and snapshot data without adding new routes.\n\n"
            "## Open Questions\n- None.\n"
        ),
        repo_understanding=repo_understanding,
        draft_phase="planner",
        min_grounding_ratio=0.45,
        requirements_text=(
            "Add a compact diagnostics dropdown to the top ActionBar that shows snapshot data, "
            "reuses existing snapshot query data from PlanningView and diagnostics backend endpoints, "
            "keeps the current PlanPanel health block intact, and adds tests."
        ),
    )

    assert not result.ok
    assert "missing_helper_reuse" in result.reason_codes
    assert any("missing explicit helper reuse reference for snapshot" in failure for failure in result.failure_messages)
    assert any("getSessionSnapshot" in failure for failure in result.failure_messages)


@pytest.mark.asyncio
async def test_run_initial_draft_passes_raw_requirements_to_planner_pipeline(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    captured: dict[str, object] = {}

    async def _fake_run_planner_pipeline(**kwargs: object) -> AuthorResult:
        captured.update(kwargs)
        return AuthorResult(plan="# Plan\n\n## Summary\nok\n", unverified_references=set(), accessed_paths=set())

    agent._run_planner_pipeline = _fake_run_planner_pipeline  # type: ignore[method-assign]

    requirements = "Add a compact diagnostics dropdown without observability work."
    await agent.run_initial_draft(
        requirements=requirements,
        manifesto="monitoring exists elsewhere in the repo manifesto",
        manifesto_path=".prscope/manifesto.md",
        skills_block="",
        recall_block="",
        context_index="",
        grounding_paths=set(),
    )

    assert captured["requirements"] == requirements


@pytest.mark.asyncio
async def test_self_review_draft_caps_hints_and_uses_evidence_bundle(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    captured: dict[str, object] = {}

    async def _fake_run_stage(stage: str, messages: list[dict[str, object]], **kwargs: object) -> str:
        captured["stage"] = stage
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return json.dumps(
            {
                "revision_hints": [
                    "Reuse `src/prscope/web/frontend/src/lib/api.ts`.",
                    "Add the adjacent planning page test target.",
                    "Do not add new endpoints.",
                    "Keep the change localized to existing UI wiring.",
                    "This fifth hint should be dropped.",
                ]
            }
        )

    agent.stage_runner.run_stage = _fake_run_stage  # type: ignore[method-assign]
    hints = await agent.self_review_draft(
        requirements="Move export controls into the top ActionBar and reuse existing helpers.",
        plan_content="# Draft\n\n## Files Changed\n- `src/prscope/web/frontend/src/components/ActionBar.tsx`\n",
        evidence_bundle=EvidenceBundle(
            relevant_files=("src/prscope/web/frontend/src/components/ActionBar.tsx",),
            existing_components=("ActionBar", "exportSession"),
            test_targets=("src/prscope/web/frontend/src/pages/PlanningView.test.ts",),
        ),
        validation_result=ValidationResult(
            failure_messages=("missing test target", "unknown file references: src/prscope/web/lib/api.ts"),
            reason_codes=("missing_tests", "unknown_file_reference"),
            retryable=True,
            failure_count=2,
        ),
        attempt_context=AttemptContext(attempt_number=2, previous_failures=("missing_tests",), elapsed_ms=1200),
    )

    assert len(hints) == 4
    assert hints[0] == "Reuse `src/prscope/web/frontend/src/lib/api.ts`."
    messages = captured["messages"]
    assert isinstance(messages, list)
    user_prompt = str(messages[1]["content"])
    assert "Structured Evidence" in user_prompt
    assert "ActionBar.tsx" in user_prompt
    assert "PlanningView.test.ts" in user_prompt


@pytest.mark.asyncio
async def test_author_planner_pipeline_allows_extra_redraft_for_localized_ui_scope_drift() -> None:
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        core_modules=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        relevant_modules=[
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/components/ActionBar.tsx",
            "src/prscope/web/frontend/src/lib/api.ts",
        ],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="frontend action bar export flow",
        risks=[],
        file_contents={"src/prscope/web/frontend/src/pages/PlanningView.tsx": "export"},
        from_mental_model=False,
    )
    calls: list[dict[str, object]] = []

    async def _draft_plan(**kwargs: object) -> str:
        calls.append(kwargs)
        if len(calls) == 1:
            return (
                "# Draft\n\n## Summary\nx\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Changes\n- z\n\n"
                "## Files Changed\n- `src/prscope/web/frontend/src/components/ActionBar.tsx`\n\n"
                "## Architecture\nObservability remains unchanged through existing logging.\n\n"
                "## Open Questions\n- None.\n"
            )
        return (
            "# Draft\n\n## Summary\nx\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Changes\n- z\n\n"
            "## Files Changed\n- `src/prscope/web/frontend/src/components/ActionBar.tsx`\n\n"
            "## Architecture\nReuse the existing UI wiring and API helpers without adding new architecture layers.\n\n"
            "## Open Questions\n- None.\n"
        )

    pipeline = AuthorPlannerPipeline(
        tool_executor=SimpleNamespace(memory_block_callback=None, read_history={}),
        scan_repo_candidates=lambda **_: RepoCandidates(
            entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
            source_modules=[
                "src/prscope/web/frontend/src/pages/PlanningView.tsx",
                "src/prscope/web/frontend/src/components/ActionBar.tsx",
                "src/prscope/web/frontend/src/lib/api.ts",
            ],
            tests_and_config=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
            all_paths=[
                "src/prscope/web/frontend/src/pages/PlanningView.tsx",
                "src/prscope/web/frontend/src/components/ActionBar.tsx",
                "src/prscope/web/frontend/src/lib/api.ts",
                "src/prscope/web/frontend/src/pages/PlanningView.test.ts",
            ],
        ),
        explore_repo=lambda **_: repo_understanding,
        classify_complexity=lambda **_: "moderate",
        draft_plan=_draft_plan,
        validate_draft=AuthorAgent(  # type: ignore[misc]
            config=PlanningConfig(author_model="gpt-4o-mini"),
            tool_executor=ToolExecutor(Path(".")),
        ).validate_draft_result,
    )

    result = await pipeline.run(
        requirements=(
            "Move export controls into the top ActionBar and reuse existing backend endpoints and frontend export helpers."
        ),
        min_grounding_ratio=0.35,
        grounding_paths={"src/prscope/web/frontend/src/components/ActionBar.tsx"},
        model_override=None,
        rejection_counts={},
        rejection_reasons=[],
        timeout_seconds_override=None,
    )

    assert len(calls) == 2
    assert "Observability" not in result.plan


@pytest.mark.asyncio
async def test_author_planner_pipeline_reuses_same_evidence_and_keeps_best_draft() -> None:
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        core_modules=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        relevant_modules=[
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/components/ActionBar.tsx",
            "src/prscope/web/frontend/src/lib/api.ts",
        ],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="frontend action bar export flow",
        risks=[],
        file_contents={
            "src/prscope/web/frontend/src/pages/PlanningView.tsx": "export function PlanningView() {}",
            "src/prscope/web/frontend/src/components/ActionBar.tsx": "export function ActionBar() {}",
            "src/prscope/web/frontend/src/lib/api.ts": "export function exportSession() {}",
        },
        from_mental_model=False,
    )
    draft_calls: list[dict[str, object]] = []
    explore_calls = 0
    scan_calls = 0

    async def _draft_plan(**kwargs: object) -> str:
        draft_calls.append(kwargs)
        if len(draft_calls) == 1:
            return (
                "# Draft\n\n## Summary\nx\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Changes\n- z\n\n"
                "## Files Changed\n- `src/prscope/web/frontend/src/components/ActionBar.tsx`\n\n"
                "## Architecture\nKeep the change localized.\n\n## Open Questions\n- None.\n"
            )
        return (
            "# Draft\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Files Changed\n- `src/unknown.py`\n\n"
            "## Architecture\nObservability logging.\n"
        )

    async def _self_review(**_: object) -> list[str]:
        return ["Keep the original grounded files and add the missing summary only."]

    def _scan_repo_candidates(**_: object) -> RepoCandidates:
        nonlocal scan_calls
        scan_calls += 1
        return RepoCandidates(
            entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
            source_modules=[
                "src/prscope/web/frontend/src/pages/PlanningView.tsx",
                "src/prscope/web/frontend/src/components/ActionBar.tsx",
                "src/prscope/web/frontend/src/lib/api.ts",
            ],
            tests_and_config=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
            all_paths=[
                "src/prscope/web/frontend/src/pages/PlanningView.tsx",
                "src/prscope/web/frontend/src/components/ActionBar.tsx",
                "src/prscope/web/frontend/src/lib/api.ts",
                "src/prscope/web/frontend/src/pages/PlanningView.test.ts",
            ],
        )

    def _explore_repo(**_: object) -> RepoUnderstanding:
        nonlocal explore_calls
        explore_calls += 1
        return repo_understanding

    validation_results = [
        ValidationResult(
            failure_messages=("missing summary",),
            reason_codes=("missing_sections",),
            retryable=True,
            failure_count=1,
        ),
        ValidationResult(
            failure_messages=("missing summary", "unknown file references: src/unknown.py"),
            reason_codes=("missing_sections", "unknown_file_reference"),
            retryable=True,
            failure_count=2,
        ),
    ]

    pipeline = AuthorPlannerPipeline(
        tool_executor=SimpleNamespace(memory_block_callback=None, read_history={}),
        scan_repo_candidates=_scan_repo_candidates,
        explore_repo=_explore_repo,
        classify_complexity=lambda **_: "moderate",
        draft_plan=_draft_plan,
        validate_draft=lambda **_: validation_results.pop(0),
        self_review_draft=_self_review,
    )

    result = await pipeline.run(
        requirements="Move export controls into the top ActionBar and reuse existing backend endpoints and helpers.",
        min_grounding_ratio=0.35,
        grounding_paths={"src/prscope/web/frontend/src/components/ActionBar.tsx"},
        model_override=None,
        rejection_counts={},
        rejection_reasons=[],
        timeout_seconds_override=None,
    )

    assert scan_calls == 1
    assert explore_calls == 1
    assert len(draft_calls) == 2
    assert draft_calls[0]["evidence_bundle"] == draft_calls[1]["evidence_bundle"]
    assert "src/prscope/web/frontend/src/components/ActionBar.tsx" in result.plan
    assert "src/unknown.py" not in result.plan
    assert result.draft_diagnostics["author_internal_attempts"] == 2
    assert result.draft_diagnostics["author_self_review_used"] is True


@pytest.mark.asyncio
async def test_author_planner_pipeline_allows_retry_after_slow_first_attempt_for_moderate_complexity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        core_modules=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        relevant_modules=[
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/components/ActionBar.tsx",
            "src/prscope/web/frontend/src/lib/api.ts",
        ],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="frontend action bar export flow",
        risks=[],
        file_contents={
            "src/prscope/web/frontend/src/pages/PlanningView.tsx": "export function PlanningViewPage() {}",
            "src/prscope/web/frontend/src/components/ActionBar.tsx": "export function ActionBar() {}",
            "src/prscope/web/frontend/src/lib/api.ts": "export function exportSession() {}",
        },
        from_mental_model=False,
    )
    draft_calls: list[dict[str, object]] = []

    async def _draft_plan(**kwargs: object) -> str:
        draft_calls.append(kwargs)
        if len(draft_calls) == 1:
            return (
                "# Draft\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Files Changed\n"
                "- `src/prscope/web/frontend/src/components/ActionBar.tsx`\n\n"
                "## Architecture\nObservability logging remains unchanged.\n"
            )
        return (
            "# Draft\n\n## Summary\nx\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Changes\n- z\n\n"
            "## Files Changed\n- `src/prscope/web/frontend/src/components/ActionBar.tsx`\n"
            "- `src/prscope/web/frontend/src/pages/PlanningView.test.ts`\n\n"
            "## Architecture\nReuse `exportSession` without extra observability work.\n\n"
            "## Open Questions\n- None.\n"
        )

    validation_results = [
        ValidationResult(
            failure_messages=("missing test target reference; reference one of: src/prscope/web/frontend/src/pages/PlanningView.test.ts",),
            reason_codes=("missing_tests",),
            retryable=True,
            failure_count=1,
        ),
        ValidationResult.success(),
        ValidationResult.success(),
    ]

    class _FakePerfCounter:
        def __init__(self, values: list[float]) -> None:
            self._values = iter(values)
            self._last = values[-1]

        def __call__(self) -> float:
            try:
                self._last = next(self._values)
            except StopIteration:
                pass
            return self._last

    monkeypatch.setattr(
        pipeline_module.time,
        "perf_counter",
        _FakePerfCounter([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 8.8, 8.9, 9.0, 9.1, 11.0, 11.1, 11.2]),
    )

    pipeline = AuthorPlannerPipeline(
        tool_executor=SimpleNamespace(memory_block_callback=None, read_history={}),
        scan_repo_candidates=lambda **_: RepoCandidates(
            entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
            source_modules=[
                "src/prscope/web/frontend/src/pages/PlanningView.tsx",
                "src/prscope/web/frontend/src/components/ActionBar.tsx",
                "src/prscope/web/frontend/src/lib/api.ts",
            ],
            tests_and_config=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
            all_paths=[
                "src/prscope/web/frontend/src/pages/PlanningView.tsx",
                "src/prscope/web/frontend/src/components/ActionBar.tsx",
                "src/prscope/web/frontend/src/lib/api.ts",
                "src/prscope/web/frontend/src/pages/PlanningView.test.ts",
            ],
        ),
        explore_repo=lambda **_: repo_understanding,
        classify_complexity=lambda **_: "moderate",
        draft_plan=_draft_plan,
        validate_draft=lambda **_: validation_results.pop(0),
    )

    result = await pipeline.run(
        requirements="Move export controls into the top ActionBar and add tests.",
        min_grounding_ratio=0.35,
        grounding_paths={"src/prscope/web/frontend/src/components/ActionBar.tsx"},
        model_override=None,
        rejection_counts={},
        rejection_reasons=[],
        timeout_seconds_override=None,
    )

    assert len(draft_calls) == 2
    assert result.draft_diagnostics["author_internal_attempts"] == 2
    assert result.draft_diagnostics["draft_loop_budget_ms"] == 16000


@pytest.mark.asyncio
async def test_author_planner_pipeline_allows_retry_after_slow_first_attempt_for_complex_complexity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        core_modules=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        relevant_modules=[
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/components/ActionBar.tsx",
            "src/prscope/web/frontend/src/lib/api.ts",
        ],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="frontend diagnostics dropdown flow",
        risks=[],
        file_contents={
            "src/prscope/web/frontend/src/pages/PlanningView.tsx": "export function PlanningViewPage() {}",
            "src/prscope/web/frontend/src/components/ActionBar.tsx": "export function ActionBar() {}",
            "src/prscope/web/frontend/src/lib/api.ts": "export function getSessionSnapshot() {}",
        },
        from_mental_model=False,
    )
    draft_calls: list[dict[str, object]] = []

    async def _draft_plan(**kwargs: object) -> str:
        draft_calls.append(kwargs)
        if len(draft_calls) == 1:
            return (
                "# Draft\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Files Changed\n"
                "- `src/prscope/web/frontend/src/components/ActionBar.tsx`\n"
                "- `src/prscope/web/frontend/src/lib/api.ts`\n\n"
                "## Architecture\nReuse `getSessionSnapshot` without adding routes.\n"
            )
        return (
            "# Draft\n\n## Summary\nx\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Changes\n- z\n\n"
            "## Files Changed\n- `src/prscope/web/frontend/src/components/ActionBar.tsx`\n"
            "- `src/prscope/web/frontend/src/lib/api.ts`\n"
            "- `src/prscope/web/frontend/src/pages/PlanningView.test.ts`\n\n"
            "## Architecture\nReuse `getSessionSnapshot` without adding routes.\n\n"
            "## Open Questions\n- None.\n"
        )

    validation_results = [
        ValidationResult(
            failure_messages=("missing test target reference; reference one of: src/prscope/web/frontend/src/pages/PlanningView.test.ts",),
            reason_codes=("missing_tests",),
            retryable=True,
            failure_count=1,
        ),
        ValidationResult.success(),
        ValidationResult.success(),
    ]

    class _FakePerfCounter:
        def __init__(self, values: list[float]) -> None:
            self._values = iter(values)
            self._last = values[-1]

        def __call__(self) -> float:
            try:
                self._last = next(self._values)
            except StopIteration:
                pass
            return self._last

    monkeypatch.setattr(
        pipeline_module.time,
        "perf_counter",
        _FakePerfCounter([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 24.0, 24.1, 24.2, 24.3, 26.4, 26.5, 26.6]),
    )

    pipeline = AuthorPlannerPipeline(
        tool_executor=SimpleNamespace(memory_block_callback=None, read_history={}),
        scan_repo_candidates=lambda **_: RepoCandidates(
            entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
            source_modules=[
                "src/prscope/web/frontend/src/pages/PlanningView.tsx",
                "src/prscope/web/frontend/src/components/ActionBar.tsx",
                "src/prscope/web/frontend/src/lib/api.ts",
            ],
            tests_and_config=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
            all_paths=[
                "src/prscope/web/frontend/src/pages/PlanningView.tsx",
                "src/prscope/web/frontend/src/components/ActionBar.tsx",
                "src/prscope/web/frontend/src/lib/api.ts",
                "src/prscope/web/frontend/src/pages/PlanningView.test.ts",
            ],
        ),
        explore_repo=lambda **_: repo_understanding,
        classify_complexity=lambda **_: "complex",
        draft_plan=_draft_plan,
        validate_draft=lambda **_: validation_results.pop(0),
    )

    result = await pipeline.run(
        requirements="Add a compact diagnostics dropdown and add tests.",
        min_grounding_ratio=0.45,
        grounding_paths={"src/prscope/web/frontend/src/components/ActionBar.tsx"},
        model_override=None,
        rejection_counts={},
        rejection_reasons=[],
        timeout_seconds_override=None,
    )

    assert len(draft_calls) == 2
    assert result.draft_diagnostics["author_internal_attempts"] == 2
    assert result.draft_diagnostics["draft_loop_budget_ms"] == 28000


@pytest.mark.asyncio
async def test_author_planner_pipeline_returns_best_draft_when_retry_attempt_raises() -> None:
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        core_modules=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        relevant_modules=[
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/components/ActionBar.tsx",
            "src/prscope/web/frontend/src/lib/api.ts",
        ],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="frontend diagnostics dropdown flow",
        risks=[],
        file_contents={
            "src/prscope/web/frontend/src/pages/PlanningView.tsx": "export function PlanningViewPage() {}",
            "src/prscope/web/frontend/src/components/ActionBar.tsx": "export function ActionBar() {}",
            "src/prscope/web/frontend/src/lib/api.ts": "export function getSessionSnapshot() {}",
        },
        from_mental_model=False,
    )
    draft_calls = 0

    async def _draft_plan(**_: object) -> str:
        nonlocal draft_calls
        draft_calls += 1
        if draft_calls == 1:
            return (
                "# Draft\n\n## Summary\nx\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Changes\n- z\n\n"
                "## Files Changed\n- `src/prscope/web/frontend/src/components/ActionBar.tsx`\n"
                "- `src/prscope/web/frontend/src/pages/PlanningView.test.ts`\n\n"
                "## Architecture\nReuse `getSessionSnapshot`.\n\n## Open Questions\n- None.\n"
            )
        raise TimeoutError("second attempt timed out")

    pipeline = AuthorPlannerPipeline(
        tool_executor=SimpleNamespace(memory_block_callback=None, read_history={}),
        scan_repo_candidates=lambda **_: RepoCandidates(
            entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
            source_modules=[
                "src/prscope/web/frontend/src/pages/PlanningView.tsx",
                "src/prscope/web/frontend/src/components/ActionBar.tsx",
                "src/prscope/web/frontend/src/lib/api.ts",
            ],
            tests_and_config=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
            all_paths=[
                "src/prscope/web/frontend/src/pages/PlanningView.tsx",
                "src/prscope/web/frontend/src/components/ActionBar.tsx",
                "src/prscope/web/frontend/src/lib/api.ts",
                "src/prscope/web/frontend/src/pages/PlanningView.test.ts",
            ],
        ),
        explore_repo=lambda **_: repo_understanding,
        classify_complexity=lambda **_: "complex",
        draft_plan=_draft_plan,
        validate_draft=lambda **_: ValidationResult(
            failure_messages=("missing test target reference; reference one of: src/prscope/web/frontend/src/pages/PlanningView.test.ts",),
            reason_codes=("missing_tests",),
            retryable=True,
            failure_count=1,
        ),
    )

    result = await pipeline.run(
        requirements="Add a compact diagnostics dropdown and add tests.",
        min_grounding_ratio=0.45,
        grounding_paths={"src/prscope/web/frontend/src/components/ActionBar.tsx"},
        model_override=None,
        rejection_counts={},
        rejection_reasons=[],
        timeout_seconds_override=None,
    )

    assert draft_calls == 2
    assert "getSessionSnapshot" in result.plan
    assert result.draft_diagnostics["retry_exception"] == "attempt 2 failed after best draft: second attempt timed out"


@pytest.mark.asyncio
async def test_author_planner_pipeline_deterministically_repairs_missing_test_target_reference() -> None:
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        core_modules=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        relevant_modules=[
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/components/ActionBar.tsx",
            "src/prscope/web/frontend/src/lib/api.ts",
        ],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="frontend diagnostics dropdown flow",
        risks=[],
        file_contents={
            "src/prscope/web/frontend/src/pages/PlanningView.tsx": "export function PlanningViewPage() {}",
            "src/prscope/web/frontend/src/components/ActionBar.tsx": "export function ActionBar() {}",
            "src/prscope/web/frontend/src/lib/api.ts": "export function getSessionSnapshot() {}",
        },
        from_mental_model=False,
    )
    validations = [
        ValidationResult(
            failure_messages=("missing test target reference; reference one of: src/prscope/web/frontend/src/pages/PlanningView.test.ts",),
            reason_codes=("missing_tests",),
            retryable=True,
            failure_count=1,
        ),
        ValidationResult(
            failure_messages=("missing test target reference; reference one of: src/prscope/web/frontend/src/pages/PlanningView.test.ts",),
            reason_codes=("missing_tests",),
            retryable=True,
            failure_count=1,
        ),
        ValidationResult.success(),
    ]

    async def _draft_plan(**_: object) -> str:
        return (
            "# Draft\n\n## Summary\nx\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Changes\n- z\n\n"
            "## Files Changed\n- `src/prscope/web/frontend/src/components/ActionBar.tsx`\n"
            "- `src/prscope/web/frontend/src/lib/api.ts`\n\n"
            "## Architecture\nReuse `getSessionSnapshot`.\n\n## Open Questions\n- None.\n"
        )

    pipeline = AuthorPlannerPipeline(
        tool_executor=SimpleNamespace(memory_block_callback=None, read_history={}),
        scan_repo_candidates=lambda **_: RepoCandidates(
            entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
            source_modules=[
                "src/prscope/web/frontend/src/pages/PlanningView.tsx",
                "src/prscope/web/frontend/src/components/ActionBar.tsx",
                "src/prscope/web/frontend/src/lib/api.ts",
            ],
            tests_and_config=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
            all_paths=[
                "src/prscope/web/frontend/src/pages/PlanningView.tsx",
                "src/prscope/web/frontend/src/components/ActionBar.tsx",
                "src/prscope/web/frontend/src/lib/api.ts",
                "src/prscope/web/frontend/src/pages/PlanningView.test.ts",
            ],
        ),
        explore_repo=lambda **_: repo_understanding,
        classify_complexity=lambda **_: "complex",
        draft_plan=_draft_plan,
        validate_draft=lambda **_: validations.pop(0),
    )

    result = await pipeline.run(
        requirements="Add a compact diagnostics dropdown and add tests.",
        min_grounding_ratio=0.45,
        grounding_paths={"src/prscope/web/frontend/src/components/ActionBar.tsx"},
        model_override=None,
        rejection_counts={},
        rejection_reasons=[],
        timeout_seconds_override=None,
    )

    assert "src/prscope/web/frontend/src/pages/PlanningView.test.ts" in result.plan
    assert result.draft_diagnostics["quality_gate_failures"] == []


@pytest.mark.asyncio
async def test_author_planner_pipeline_deterministically_repairs_localized_scope_drift() -> None:
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        core_modules=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        relevant_modules=[
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/components/ActionBar.tsx",
            "src/prscope/web/frontend/src/lib/api.ts",
        ],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="frontend export action flow",
        risks=[],
        file_contents={
            "src/prscope/web/frontend/src/pages/PlanningView.tsx": "export function PlanningViewPage() {}",
            "src/prscope/web/frontend/src/components/ActionBar.tsx": "export function ActionBar() {}",
            "src/prscope/web/frontend/src/lib/api.ts": "export function exportSession() {}\nexport function downloadFile() {}",
        },
        from_mental_model=False,
    )
    validations = [
        ValidationResult(
            failure_messages=(
                "localized UI/API draft introduced observability or telemetry scope not present in requirements; "
                "remove logging/telemetry/observability wording and keep the plan focused on existing UI wiring, "
                "API helper reuse, and tests",
            ),
            reason_codes=("localized_scope_drift",),
            retryable=True,
            failure_count=1,
        ),
        ValidationResult(
            failure_messages=(
                "localized UI/API draft introduced observability or telemetry scope not present in requirements; "
                "remove logging/telemetry/observability wording and keep the plan focused on existing UI wiring, "
                "API helper reuse, and tests",
            ),
            reason_codes=("localized_scope_drift",),
            retryable=True,
            failure_count=1,
        ),
        ValidationResult.success(),
    ]

    async def _draft_plan(**_: object) -> str:
        return (
            "# Draft\n\n## Summary\nx\n\n## Goals\n- x\n\n## Non-Goals\n- Avoid introducing observability work.\n\n"
            "## Changes\n- z\n\n## Files Changed\n- `src/prscope/web/frontend/src/components/ActionBar.tsx`\n"
            "- `src/prscope/web/frontend/src/lib/api.ts`\n"
            "- `src/prscope/web/frontend/src/pages/PlanningView.test.ts`\n\n"
            "## Architecture\nReuse `exportSession` and `downloadFile` without logging changes.\n\n"
            "## Open Questions\n- None.\n"
        )

    pipeline = AuthorPlannerPipeline(
        tool_executor=SimpleNamespace(memory_block_callback=None, read_history={}),
        scan_repo_candidates=lambda **_: RepoCandidates(
            entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
            source_modules=[
                "src/prscope/web/frontend/src/pages/PlanningView.tsx",
                "src/prscope/web/frontend/src/components/ActionBar.tsx",
                "src/prscope/web/frontend/src/lib/api.ts",
            ],
            tests_and_config=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
            all_paths=[
                "src/prscope/web/frontend/src/pages/PlanningView.tsx",
                "src/prscope/web/frontend/src/components/ActionBar.tsx",
                "src/prscope/web/frontend/src/lib/api.ts",
                "src/prscope/web/frontend/src/pages/PlanningView.test.ts",
            ],
        ),
        explore_repo=lambda **_: repo_understanding,
        classify_complexity=lambda **_: "moderate",
        draft_plan=_draft_plan,
        validate_draft=lambda **_: validations.pop(0),
    )

    result = await pipeline.run(
        requirements="Move export controls into the ActionBar and add tests.",
        min_grounding_ratio=0.45,
        grounding_paths={"src/prscope/web/frontend/src/components/ActionBar.tsx"},
        model_override=None,
        rejection_counts={},
        rejection_reasons=[],
        timeout_seconds_override=None,
    )

    assert "observability" not in result.plan.lower()
    assert "logging" not in result.plan.lower()
    assert result.draft_diagnostics["quality_gate_failures"] == []


@pytest.mark.asyncio
async def test_author_planner_pipeline_preserves_helper_owner_file_when_helper_reuse_is_explicit() -> None:
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        core_modules=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        relevant_modules=[
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/components/ActionBar.tsx",
            "src/prscope/web/frontend/src/lib/api.ts",
        ],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="frontend export action flow",
        risks=[],
        file_contents={
            "src/prscope/web/frontend/src/pages/PlanningView.tsx": "export function PlanningViewPage() {}",
            "src/prscope/web/frontend/src/components/ActionBar.tsx": "export function ActionBar() {}",
            "src/prscope/web/frontend/src/lib/api.ts": "export function exportSession() {}\nexport function downloadFile() {}",
        },
        from_mental_model=False,
    )
    validations = [
        ValidationResult(
            failure_messages=("missing test target reference; reference one of: src/prscope/web/frontend/src/pages/PlanningView.test.ts",),
            reason_codes=("missing_tests",),
            retryable=True,
            failure_count=1,
        ),
        ValidationResult.success(),
        ValidationResult.success(),
    ]

    async def _draft_plan(**_: object) -> str:
        return (
            "# Draft\n\n## Summary\nx\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Changes\n"
            "- Reuse `exportSession` and `downloadFile`.\n\n"
            "## Files Changed\n- `src/prscope/web/frontend/src/components/ActionBar.tsx`\n"
            "- `src/prscope/web/frontend/src/pages/PlanningView.test.ts`\n\n"
            "## Architecture\nReuse `exportSession` and `downloadFile` from the existing helper layer.\n\n"
            "## Open Questions\n- None.\n"
        )

    pipeline = AuthorPlannerPipeline(
        tool_executor=SimpleNamespace(memory_block_callback=None, read_history={}),
        scan_repo_candidates=lambda **_: RepoCandidates(
            entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
            source_modules=[
                "src/prscope/web/frontend/src/pages/PlanningView.tsx",
                "src/prscope/web/frontend/src/components/ActionBar.tsx",
                "src/prscope/web/frontend/src/lib/api.ts",
            ],
            tests_and_config=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
            all_paths=[
                "src/prscope/web/frontend/src/pages/PlanningView.tsx",
                "src/prscope/web/frontend/src/components/ActionBar.tsx",
                "src/prscope/web/frontend/src/lib/api.ts",
                "src/prscope/web/frontend/src/pages/PlanningView.test.ts",
            ],
        ),
        explore_repo=lambda **_: repo_understanding,
        classify_complexity=lambda **_: "moderate",
        draft_plan=_draft_plan,
        validate_draft=lambda **_: validations.pop(0),
    )

    result = await pipeline.run(
        requirements="Move export controls into the ActionBar and add tests.",
        min_grounding_ratio=0.45,
        grounding_paths={"src/prscope/web/frontend/src/components/ActionBar.tsx"},
        model_override=None,
        rejection_counts={},
        rejection_reasons=[],
        timeout_seconds_override=None,
    )

    assert "src/prscope/web/frontend/src/lib/api.ts" in result.plan


@pytest.mark.asyncio
async def test_author_planner_pipeline_enriches_evidence_from_targeted_frontend_reads() -> None:
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        core_modules=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        relevant_modules=[
            "src/prscope/web/frontend/src/components/ActionBar.tsx",
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/lib/api.ts",
        ],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="frontend action bar export flow",
        risks=[],
        file_contents={
            "src/prscope/web/frontend/src/pages/PlanningView.tsx": "queryFn: () => getSessionSnapshot(id)",
            "src/prscope/web/frontend/src/lib/api.ts": "return request({})",
            "src/prscope/web/frontend/src/components/ActionBar.tsx": "export function ActionBar() {}",
        },
        from_mental_model=False,
    )
    tool_executor = SimpleNamespace(
        memory_block_callback=None,
        read_history={},
        read_file=lambda path, max_lines=200: {
            "content": (
                "import {\n  downloadFile,\n  exportSession,\n  getSessionSnapshot,\n} from '../lib/api';\n"
                if path.endswith("PlanningView.tsx")
                else "export function getSessionSnapshot() {}\nexport function exportSession() {}\nexport function downloadFile() {}\n"
            )
        },
    )
    captured: list[dict[str, object]] = []

    async def _draft_plan(**kwargs: object) -> str:
        captured.append(kwargs)
        return (
            "# Draft\n\n## Summary\nx\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Changes\n- z\n\n"
            "## Files Changed\n- `src/prscope/web/frontend/src/components/ActionBar.tsx`\n\n"
            "## Architecture\nReuse `exportSession` and `downloadFile`.\n\n## Open Questions\n- None.\n"
        )

    pipeline = AuthorPlannerPipeline(
        tool_executor=tool_executor,
        scan_repo_candidates=lambda **_: RepoCandidates(
            entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
            source_modules=[
                "src/prscope/web/frontend/src/pages/PlanningView.tsx",
                "src/prscope/web/frontend/src/components/ActionBar.tsx",
                "src/prscope/web/frontend/src/lib/api.ts",
            ],
            tests_and_config=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
            all_paths=[
                "src/prscope/web/frontend/src/pages/PlanningView.tsx",
                "src/prscope/web/frontend/src/components/ActionBar.tsx",
                "src/prscope/web/frontend/src/lib/api.ts",
                "src/prscope/web/frontend/src/pages/PlanningView.test.ts",
            ],
        ),
        explore_repo=lambda **_: repo_understanding,
        classify_complexity=lambda **_: "simple",
        draft_plan=_draft_plan,
        validate_draft=lambda **_: ValidationResult.success(),
    )

    await pipeline.run(
        requirements=(
            "Move export controls into the top ActionBar and reuse existing backend export/download endpoints and frontend export helpers."
        ),
        min_grounding_ratio=None,
        grounding_paths={"src/prscope/web/frontend/src/components/ActionBar.tsx"},
        model_override=None,
        rejection_counts={},
        rejection_reasons=[],
        timeout_seconds_override=None,
    )

    evidence_bundle = captured[0]["evidence_bundle"]
    assert isinstance(evidence_bundle, EvidenceBundle)
    assert "exportSession" in evidence_bundle.existing_components
    assert "downloadFile" in evidence_bundle.existing_components
    assert "getSessionSnapshot" in evidence_bundle.existing_routes_or_helpers


@pytest.mark.asyncio
async def test_author_planner_pipeline_records_stability_stop_for_same_failure_signature() -> None:
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/api.py"],
        core_modules=["src/prscope/web/api.py"],
        relevant_modules=["src/prscope/web/api.py"],
        relevant_tests=["tests/test_web_api_models.py"],
        architecture_summary="single FastAPI module",
        risks=[],
        file_contents={"src/prscope/web/api.py": "@app.get('/health')\nasync def health():\n    return {'status': 'ok'}\n"},
        from_mental_model=False,
    )
    draft_calls = 0

    async def _draft_plan(**_: object) -> str:
        nonlocal draft_calls
        draft_calls += 1
        return "# Draft\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Files Changed\n- `src/prscope/web/api.py`\n\n## Architecture\n- z\n"

    async def _self_review(**_: object) -> list[str]:
        return ["Add the missing summary section."]

    pipeline = AuthorPlannerPipeline(
        tool_executor=SimpleNamespace(memory_block_callback=None, read_history={}),
        scan_repo_candidates=lambda **_: RepoCandidates(
            entrypoints=["src/prscope/web/api.py"],
            source_modules=["src/prscope/web/api.py"],
            tests_and_config=["tests/test_web_api_models.py"],
            all_paths=["src/prscope/web/api.py", "tests/test_web_api_models.py"],
        ),
        explore_repo=lambda **_: repo_understanding,
        classify_complexity=lambda **_: "moderate",
        draft_plan=_draft_plan,
        validate_draft=lambda **_: ValidationResult(
            failure_messages=("missing summary",),
            reason_codes=("missing_sections",),
            retryable=True,
            failure_count=1,
        ),
        self_review_draft=_self_review,
    )

    result = await pipeline.run(
        requirements="Enhance the existing health implementation with observability.",
        min_grounding_ratio=0.35,
        grounding_paths={"src/prscope/web/api.py"},
        model_override=None,
        rejection_counts={},
        rejection_reasons=[],
        timeout_seconds_override=None,
    )

    assert draft_calls == 2
    assert result.draft_diagnostics["stability_stop"] is True
    assert result.draft_diagnostics["stability_reason_codes"] == ["missing_sections"]


@pytest.mark.asyncio
async def test_author_planner_pipeline_run_logs_without_formatting_errors(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    repo_understanding = RepoUnderstanding(
        entrypoints=["app.py"],
        core_modules=["app.py"],
        relevant_modules=["app.py"],
        relevant_tests=["tests/test_app.py"],
        architecture_summary="single module",
        risks=[],
        file_contents={"app.py": "def health():\n    return {'status': 'healthy'}\n"},
        from_mental_model=False,
    )

    async def _draft_plan(**_: object) -> str:
        return "Update `app.py` and `tests/test_app.py`."

    pipeline = AuthorPlannerPipeline(
        tool_executor=SimpleNamespace(memory_block_callback=None, read_history={"app.py": "read"}),
        scan_repo_candidates=lambda **_: RepoCandidates(
            entrypoints=["app.py"],
            source_modules=["app.py"],
            tests_and_config=["tests/test_app.py"],
            all_paths=["app.py", "tests/test_app.py"],
        ),
        explore_repo=lambda **_: repo_understanding,
        classify_complexity=lambda **_: "simple",
        draft_plan=_draft_plan,
        validate_draft=lambda **_: [],
    )

    result = await pipeline.run(
        requirements="Improve the health endpoint summary.",
        min_grounding_ratio=None,
        grounding_paths={"app.py"},
        model_override=None,
        rejection_counts={},
        rejection_reasons=[],
        timeout_seconds_override=None,
    )

    assert result.plan.startswith("Update `app.py`")
    assert "planner_pipeline total=" in caplog.text


@pytest.mark.asyncio
async def test_author_planner_pipeline_redrafts_invalid_planner_output() -> None:
    repo_understanding = RepoUnderstanding(
        entrypoints=["prscope/web/api.py"],
        core_modules=["prscope/web/api.py"],
        relevant_modules=["prscope/web/api.py"],
        relevant_tests=["tests/test_web_api_models.py"],
        architecture_summary="single FastAPI module",
        risks=[],
        file_contents={
            "prscope/web/api.py": "@app.get('/health')\nasync def health():\n    return {'status': 'healthy'}\n"
        },
        from_mental_model=False,
    )
    calls: list[dict[str, object]] = []

    async def _draft_plan(**kwargs: object) -> str:
        calls.append(kwargs)
        if len(calls) == 1:
            return """# Bad Planner Draft

## Goals
- Add observability.

## Non-Goals
- None.

## Files Changed
- `prscope/web/api.py`

## Architecture
- Add logs.

## Implementation Steps
1. Modify `prscope/web/api.py`.
"""
        return """# Health Observability Outline

## Summary
Add lightweight observability to the existing health route without changing its external behavior.

## Goals
- Add request logging for health checks.
- Capture simple health endpoint metrics in the existing backend.

## Non-Goals
- Redesign the health endpoint contract.

## Changes
- Extend the existing health implementation in `prscope/web/api.py`.
- Cover the new behavior in `tests/test_web_api_models.py`.

## Files Changed
- `prscope/web/api.py` to record health-check telemetry.
- `tests/test_web_api_models.py` to verify the endpoint contract and observability hooks.

## Architecture
- Keep the FastAPI route in `prscope/web/api.py` as the integration point for health telemetry.
- Emit lightweight logs/metrics alongside the existing response path so the endpoint stays cheap and deterministic.

## Open Questions
- None.
"""

    pipeline = AuthorPlannerPipeline(
        tool_executor=SimpleNamespace(memory_block_callback=None, read_history={"prscope/web/api.py": "read"}),
        scan_repo_candidates=lambda **_: RepoCandidates(
            entrypoints=["prscope/web/api.py"],
            source_modules=["prscope/web/api.py"],
            tests_and_config=["tests/test_web_api_models.py"],
            all_paths=["prscope/web/api.py", "tests/test_web_api_models.py"],
        ),
        explore_repo=lambda **_: repo_understanding,
        classify_complexity=lambda **_: "moderate",
        draft_plan=_draft_plan,
        validate_draft=AuthorAgent(  # type: ignore[misc]
            config=PlanningConfig(author_model="gpt-4o-mini"),
            tool_executor=ToolExecutor(Path(".")),
        ).validate_draft_result,
    )

    result = await pipeline.run(
        requirements="Enhance the existing health implementation with observability.",
        min_grounding_ratio=0.35,
        grounding_paths={"prscope/web/api.py"},
        model_override=None,
        rejection_counts={},
        rejection_reasons=[],
        timeout_seconds_override=None,
    )

    assert len(calls) == 2
    assert calls[0]["draft_phase"] == "planner"
    assert calls[1]["draft_phase"] == "planner"
    assert calls[1]["revision_hints"]
    assert "Implementation Steps" not in result.plan
    assert "`prscope/web/api.py`" in result.plan


@pytest.mark.asyncio
async def test_author_planner_pipeline_accepts_grounded_seed_path_without_redraft() -> None:
    repo_understanding = RepoUnderstanding(
        entrypoints=[],
        core_modules=[],
        relevant_modules=[],
        relevant_tests=[],
        architecture_summary="seeded by discovery only",
        risks=[],
        file_contents={},
        from_mental_model=False,
    )
    calls: list[dict[str, object]] = []

    async def _draft_plan(**kwargs: object) -> str:
        calls.append(kwargs)
        return """# Health Observability Outline

## Summary
Add observability to the existing health route.

## Goals
- Improve visibility into health checks.

## Non-Goals
- Create a new route.

## Changes
- Update the existing health endpoint implementation.

## Files Changed
- `prscope/web/api.py`: extend the existing health route behavior.

## Architecture
- Keep the change localized to `prscope/web/api.py`.

## Open Questions
- None.
"""

    pipeline = AuthorPlannerPipeline(
        tool_executor=SimpleNamespace(memory_block_callback=None, read_history={}),
        scan_repo_candidates=lambda **_: RepoCandidates(
            entrypoints=[],
            source_modules=[],
            tests_and_config=[],
            all_paths=[],
        ),
        explore_repo=lambda **_: repo_understanding,
        classify_complexity=lambda **_: "simple",
        draft_plan=_draft_plan,
        validate_draft=AuthorAgent(  # type: ignore[misc]
            config=PlanningConfig(author_model="gpt-4o-mini"),
            tool_executor=ToolExecutor(Path(".")),
        ).validate_draft_result,
    )

    result = await pipeline.run(
        requirements="Enhance the existing health implementation with observability.",
        min_grounding_ratio=0.35,
        grounding_paths={"prscope/web/api.py"},
        model_override=None,
        rejection_counts={},
        rejection_reasons=[],
        timeout_seconds_override=None,
    )

    assert len(calls) == 1
    assert result.rejection_reasons == []
    assert result.plan.startswith("# Health Observability Outline")


@pytest.mark.asyncio
async def test_draft_plan_uses_planner_specific_prompt(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    repo_understanding = RepoUnderstanding(
        entrypoints=["prscope/web/api.py"],
        core_modules=["prscope/web/api.py"],
        relevant_modules=["prscope/web/api.py"],
        relevant_tests=["tests/test_web_api_models.py"],
        architecture_summary="single FastAPI module",
        risks=[],
        file_contents={"prscope/web/api.py": "async def health():\n    return {'status': 'healthy'}\n"},
        from_mental_model=False,
    )
    captured: dict[str, object] = {}

    async def _fake_run_stage(stage: str, messages: list[dict[str, object]], **kwargs: object) -> str:
        captured["stage"] = stage
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return (
            "# Draft\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Files Changed\n"
            "- `prscope/web/api.py`\n\n## Architecture\n- z"
        )

    agent.stage_runner.run_stage = _fake_run_stage  # type: ignore[method-assign]

    await agent.draft_plan(
        requirements="Enhance the health endpoint with observability.",
        repo_understanding=repo_understanding,
        draft_phase="planner",
    )

    messages = captured["messages"]
    assert isinstance(messages, list)
    assert 'Do NOT include "Implementation Steps"' in str(messages[0]["content"])
    assert "Produce a concise grounded planner draft" in str(messages[1]["content"])
    assert "## Verified File Paths" in str(messages[1]["content"])
    assert "`prscope/web/api.py`" in str(messages[1]["content"])
    assert "minimal failure/error handling" in str(messages[0]["content"])
    assert "do not turn them into explicit workstreams" in str(messages[0]["content"])
    assert "do not add observability, logging, telemetry, rollout controls, or platform notes" in str(
        messages[0]["content"]
    )


@pytest.mark.asyncio
async def test_draft_plan_prioritizes_relevant_verified_paths_and_existing_guidance(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    repo_understanding = RepoUnderstanding(
        entrypoints=[f"src/module_{idx}.py" for idx in range(30)],
        core_modules=[f"src/module_{idx}.py" for idx in range(30)],
        relevant_modules=["src/prscope/web/api.py"],
        relevant_tests=["tests/test_web_api_models.py"],
        architecture_summary="single FastAPI module",
        risks=[],
        file_contents={"src/prscope/web/api.py": '@app.get("/health")\nasync def health():\n    return {"status": "healthy"}\n'},
        from_mental_model=False,
    )
    captured: dict[str, object] = {}

    async def _fake_run_stage(stage: str, messages: list[dict[str, object]], **kwargs: object) -> str:
        captured["messages"] = messages
        return (
            "# Draft\n\n## Summary\nx\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Changes\n- z\n\n"
            "## Files Changed\n- `src/prscope/web/api.py`\n\n## Architecture\n- a\n\n## Open Questions\n- None.\n"
        )

    agent.stage_runner.run_stage = _fake_run_stage  # type: ignore[method-assign]

    await agent.draft_plan(
        requirements="Add a lightweight /health endpoint and tests for it.",
        repo_understanding=repo_understanding,
        draft_phase="planner",
    )

    messages = captured["messages"]
    assert isinstance(messages, list)
    user_prompt = str(messages[1]["content"])
    assert "`src/prscope/web/api.py`" in user_prompt
    assert "`tests/test_web_api_models.py`" in user_prompt
    assert "Do not invent new test filenames or modules." in user_prompt
    assert "plan to extend that implementation rather than adding a duplicate" in user_prompt
    assert "do not describe the change as introducing a brand-new feature" in user_prompt
    assert "mention those exact helper names in the plan instead of saying only 'existing helpers'" in user_prompt
    assert "name at least one of those exact test files in the plan" in user_prompt
    assert "localized UI/API work" in user_prompt
    assert "Do not reference planning runtime or discovery modules for frontend wiring work" in user_prompt
    assert "compatibility constraint or test target" in user_prompt
    assert "do not add observability, logging, telemetry, rollout controls, or platform notes" in str(
        messages[0]["content"]
    )
