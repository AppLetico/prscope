from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prscope.config import PlanningConfig
from prscope.planning.runtime.author import AuthorAgent, RepoUnderstanding
from prscope.planning.runtime.authoring import pipeline as pipeline_module
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
from prscope.planning.runtime.authoring.pipeline import AuthorPlannerPipeline
from prscope.planning.runtime.authoring.repair import AuthorRepairService
from prscope.planning.runtime.critic import ReviewResult
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
                *[f"# filler {idx}" for idx in range(1, 190)],
                '@app.get("/health")',
                "async def health():",
                '    return {"status": "healthy"}',
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
        '@app.get("/api/sessions/{session_id}/diagnostics")\nasync def get_session_diagnostics():\n    return {}\n',
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


def test_explore_repo_includes_web_api_models_test_for_localized_backend_payload_tweak(tmp_path: Path) -> None:
    (tmp_path / "src" / "prscope" / "web" / "frontend" / "src" / "components").mkdir(parents=True)
    (tmp_path / "src" / "prscope" / "web" / "frontend" / "src" / "pages").mkdir(parents=True)
    (tmp_path / "src" / "prscope" / "web" / "frontend" / "src" / "lib").mkdir(parents=True)
    (tmp_path / "src" / "prscope" / "web").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests").mkdir()
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
        "export function getSessionSnapshot() {}\nexport function downloadFile() {}\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "prscope" / "web" / "api.py").write_text(
        "class SessionSnapshotResponse:\n    updated_at: str\n    open_issue_count: int\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "prscope" / "web" / "frontend" / "src" / "pages" / "PlanningView.test.ts").write_text(
        "test('summary chip', () => {})\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_web_api_models.py").write_text(
        "def test_snapshot_response_shape():\n    assert True\n",
        encoding="utf-8",
    )

    agent = _make_agent(tmp_path)
    candidates = agent.scan_repo_candidates(mental_model="")
    understanding = agent.explore_repo(
        requirements=(
            "Add a compact session summary chip to the ActionBar and, if a backend payload response shape tweak "
            "is needed, keep it localized to the existing web API serialization path while reusing the current "
            "snapshot/download endpoints and adding tests."
        ),
        candidates=candidates,
    )

    assert "src/prscope/web/api.py" in understanding.file_contents
    assert "tests/test_web_api_models.py" in understanding.file_contents
    assert "src/prscope/web/frontend/src/pages/PlanningView.test.ts" in understanding.file_contents


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
async def test_plan_repair_retries_with_compact_json_when_first_response_is_truncated() -> None:
    calls: list[list[dict[str, object]]] = []

    async def _fake_llm_call(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        messages = args[0] if args else None
        assert isinstance(messages, list)
        calls.append(messages)
        del kwargs
        if len(calls) == 1:
            raw = """{
  "problem_understanding": "Need to tighten the plan",
  "accepted_issues": ["scope drift"],
  "rejected_issues": ["""
        else:
            raw = """{
  "problem_understanding": "Need to tighten the plan",
  "accepted_issues": ["scope drift"],
  "rejected_issues": [],
  "root_causes": ["repair response was too verbose"],
  "repair_strategy": "Keep the fix localized",
  "target_sections": ["architecture", "implementation_steps"],
  "revision_plan": "Rewrite only the localized sections"
}"""
        return (
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw))]),
            "claude-haiku-4-5-20251001",
        )

    service = AuthorRepairService(_fake_llm_call)
    plan = PlanDocument(
        title="Plan",
        summary="Summary",
        goals="Goals",
        non_goals="Non-goals",
        files_changed="- `a.py`",
        architecture="Architecture",
        implementation_steps="1. Step",
        test_strategy="Tests",
        rollback_plan="Rollback",
        open_questions="- None.",
    )
    review = SimpleNamespace(
        architectural_concerns=["scope drift"],
        risks=[],
        recommended_changes=[],
        primary_issue="scope drift",
    )

    result = await service.plan_repair(
        review=review,
        plan=plan,
        requirements="Keep the change localized and reuse existing helpers.",
        design_record={},
        reconsideration_candidates=[],
        model_override="claude-haiku-4-5-20251001",
    )

    assert result.problem_understanding == "Need to tighten the plan"
    assert result.target_sections == ["architecture", "implementation_steps"]
    assert len(calls) == 2
    assert "Return compact JSON only" in str(calls[1][0]["content"])


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
        reconsideration_candidates=[
            {
                "decision_id": "architecture.database",
                "reason": "high_pressure_cluster",
                "decision_pressure": 6,
                "suggested_action": "reconsider architecture",
                "dominant_cluster": {
                    "root_issue_id": "issue_1",
                    "root_issue": "Database choice remains underspecified.",
                    "severity": "major",
                    "affected_plan_sections": ["architecture"],
                    "suggested_action": "reconsider architecture",
                },
            }
        ],
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
    assert "single state objects" in system_prompt
    assert "concurrency-management rhetoric" in system_prompt
    assert "code-level React prescriptions" in system_prompt
    assert "`useRef`, `useCallback`, `Promise.race`, timeout guards, or unmount guards" in system_prompt
    assert "Preserve already-grounded owner files and focused regression targets" in system_prompt
    assert "Prefer surgical edits over collapsing precise implementation or test detail" in system_prompt
    assert "If reconsideration candidates are provided" in system_prompt
    assert "Architectural Pressure Guidance" in user_prompt
    assert "Do not ignore the top pressure signal" in system_prompt or "make a visible plan change" in system_prompt
    assert "When the requirements name a source of truth" in system_prompt
    assert "manual admin actions" in system_prompt
    assert "speculative concurrency or abstraction drift" in system_prompt
    assert "separation-of-concerns/module rhetoric" in system_prompt
    assert "Preserve `src/prscope/web/frontend/src/lib/api.ts` exactly as written." in user_prompt
    assert "## Reconsideration Candidates" in user_prompt
    assert "architecture.database" in user_prompt
    assert "root issue:" in user_prompt


def test_compact_json_retry_instruction_is_strict_for_anthropic_models() -> None:
    instruction = AuthorRepairService._compact_json_retry_instruction("claude-sonnet-4-5")
    assert "Return compact JSON only." in instruction
    assert "Return exactly one JSON object and nothing after the closing brace." in instruction


@pytest.mark.asyncio
async def test_revise_plan_sanitizes_localized_frontend_state_drift_updates() -> None:
    async def _fake_llm_call(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        del args, kwargs
        raw = json.dumps(
            {
                "problem_understanding": "Tighten localized UI wiring.",
                "updates": {
                    "architecture": (
                        "**React Hook Patterns:**\n"
                        "Use a local state with a single state object, self-contained state management, hooks/contexts, "
                        "useState, useEffect, useRef, useCallback, and background polling around `exportSession`.\n"
                        "Apply state initialization, mount tracking, unmount guard, timeout recovery, and double-click prevention.\n"
                        "```typescript\n"
                        "type ExportStatus = { status: 'idle' | 'inProgress' | 'success' | 'error' };\n"
                        "const isMountedRef = useRef(true);\n"
                        "const handleExport = useCallback(async () => {\n"
                        "  await Promise.race([exportSession(sessionId), setTimeout(() => {}, 30000)]);\n"
                        "}, [sessionId]);\n"
                        "```\n"
                        "Preserve separation of concerns, separation of UI and business logic, and clear separation of state and logic."
                    ),
                    "changes": "- Add polling safeguards around the export button and manage concurrency proactively.",
                },
                "justification": {
                    "architecture": "Explains the new frontend structure.",
                    "changes": "Adds export safety.",
                },
                "review_prediction": "Reviewer should accept.",
            }
        )
        return (
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw))]),
            "gpt-4o-mini",
        )

    service = AuthorRepairService(_fake_llm_call)
    current_plan = PlanDocument(
        title="Plan",
        summary="Keep the change localized.",
        goals="Goals",
        non_goals="- Do not introduce new hooks, contexts, shared state layers, or background polling.",
        files_changed="- `src/prscope/web/frontend/src/components/PlanPanel.tsx`",
        architecture="Keep the change localized to existing `PlanPanel` wiring and helper reuse.",
        implementation_steps="1. Update the existing UI.\n2. Reuse `exportSession` and `downloadFile`.",
        test_strategy="Tests",
        rollback_plan="Rollback",
        open_questions="- None.",
    )
    repair_plan = RepairPlan(
        problem_understanding="Need to keep export wiring localized.",
        accepted_issues=["architecture drift"],
        rejected_issues=[],
        root_causes=["revision overreached"],
        repair_strategy="Tighten architecture wording",
        target_sections=["architecture", "changes"],
        revision_plan="Remove speculative frontend abstractions",
    )

    result = await service.revise_plan(
        repair_plan=repair_plan,
        current_plan=current_plan,
        requirements=(
            "Update the Planning UI to show the last export result and disable the export action while an export "
            "is already in progress. Reuse the existing exportSession and downloadFile helpers, preserve current "
            "PlanPanel behavior, and keep the change localized. Do not introduce new hooks, contexts, shared state "
            "layers, or background polling."
        ),
        revision_budget=2,
    )

    architecture = result.updates["architecture"].lower()
    changes = result.updates["changes"].lower()
    assert "local state" not in architecture
    assert "state management" not in architecture
    assert "hooks/contexts" not in architecture
    assert "usestate" not in architecture
    assert "useeffect" not in architecture
    assert "useref" not in architecture
    assert "usecallback" not in architecture
    assert "promise.race" not in architecture
    assert "settimeout" not in architecture
    assert "ismountedref" not in architecture
    assert "timeout recovery" not in architecture
    assert "mount tracking" not in architecture
    assert "double-click prevention" not in architecture
    assert "type exportstatus" not in architecture
    assert "background polling" not in architecture
    assert "separation of concerns" not in architecture
    assert "separation of ui and business logic" not in architecture
    assert "single state object" not in architecture
    assert "separation of state and logic" not in architecture
    assert "polling safeguards" not in changes
    assert "manage concurrency proactively" not in changes
    assert "localized" in architecture


def test_incremental_grounding_failures_detects_new_unverified_paths(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)

    failures = agent.incremental_grounding_failures(
        previous_plan_content=("# Plan\n\n## Files Changed\n- `src/prscope/web/frontend/src/lib/api.ts`\n"),
        updated_plan_content=(
            "# Plan\n\n## Files Changed\n- `src/prscope/web/frontend/src/lib/api.ts`\n- `src/prscope/web/lib/api.ts`\n"
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


def test_validate_draft_flags_frontend_state_abstraction_drift_for_localized_ui_requests(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    repo_understanding = RepoUnderstanding(
        entrypoints=[],
        core_modules=["src/prscope/web/frontend/src/components/PlanPanel.tsx"],
        relevant_modules=[
            "src/prscope/web/frontend/src/components/PlanPanel.tsx",
            "src/prscope/web/frontend/src/lib/api.ts",
        ],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="frontend export wiring",
        risks=[],
        file_contents={
            "src/prscope/web/frontend/src/components/PlanPanel.tsx": "export function PlanPanel() {}",
            "src/prscope/web/frontend/src/lib/api.ts": "export function exportSession() {}\nexport function downloadFile() {}",
        },
        from_mental_model=False,
    )

    failures = agent.validate_draft(
        plan_content=(
            "# Plan\n\n## Summary\nShow export result.\n\n## Goals\n- x\n\n## Non-Goals\n"
            "- Do not introduce new hooks, contexts, shared state layers, or background polling.\n\n"
            "## Changes\n- z\n\n## Files Changed\n- `src/prscope/web/frontend/src/components/PlanPanel.tsx`\n\n"
            "## Architecture\nUse a local state with a single state object, self-contained state management, hooks/contexts, useRef, Promise.race, setTimeout, and polling safeguards while preserving separation of UI and business logic and separation of state and logic.\n\n"
            "## Open Questions\n- None.\n"
        ),
        repo_understanding=repo_understanding,
        draft_phase="planner",
        requirements_text=(
            "Update the Planning UI to show the last export result and disable the export action while an export "
            "is already in progress. Reuse the existing exportSession and downloadFile helpers, preserve current "
            "PlanPanel behavior, and keep the change localized. Do not introduce new hooks, contexts, shared state "
            "layers, or background polling."
        ),
    )

    assert (
        "localized UI/API draft introduced frontend state or polling abstractions not present in requirements; "
        "remove hooks/contexts/shared-state/state-management/polling wording and keep the plan focused on direct component wiring, "
        "existing helper reuse, PlanPanel compatibility, and tests"
    ) in failures
    assert "missing explicit helper reuse reference for export; mention one of: exportSession" in failures
    assert "missing explicit helper reuse reference for download; mention one of: downloadFile" in failures


def test_validate_draft_flags_backend_contract_drift_for_localized_ui_requests(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/api.py"],
        core_modules=[
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/components/PlanPanel.tsx",
            "src/prscope/web/api.py",
        ],
        relevant_modules=[
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/components/PlanPanel.tsx",
            "src/prscope/web/frontend/src/lib/api.ts",
            "src/prscope/web/api.py",
        ],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts", "tests/test_web_api_models.py"],
        architecture_summary="frontend export wiring with existing API helpers",
        risks=[],
        file_contents={
            "src/prscope/web/frontend/src/pages/PlanningView.tsx": "export function PlanningView() {}",
            "src/prscope/web/frontend/src/components/PlanPanel.tsx": "export function PlanPanel() {}",
            "src/prscope/web/frontend/src/lib/api.ts": "export function exportSession() {}\nexport function downloadFile() {}",
            "src/prscope/web/api.py": "def get_session():\n    return {}\n",
        },
        from_mental_model=False,
    )

    failures = agent.validate_draft(
        plan_content=(
            "# Plan\n\n## Summary\nShow export result.\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n"
            "## Changes\n- Update the get_session endpoint response to include new fields `is_exporting` and "
            "`last_export_result` and sync the UI with the backend's status.\n\n"
            "## Files Changed\n- `src/prscope/web/frontend/src/pages/PlanningView.tsx`\n- `src/prscope/web/api.py`\n\n"
            "## Architecture\nAdd backend session-state plumbing via PlanningSession fields so the UI can render export "
            "status directly from the API contract.\n\n"
            "## Open Questions\n- None.\n"
        ),
        repo_understanding=repo_understanding,
        draft_phase="planner",
        requirements_text=(
            "Update the Planning UI to show the last export result and disable the export action while an export "
            "is already in progress. Reuse the existing exportSession and downloadFile helpers, preserve current "
            "PlanPanel behavior, and keep the change localized. Do not introduce new hooks, contexts, shared state "
            "layers, or background polling. Include focused frontend tests and any minimal backend contract test only "
            "if the response shape must change."
        ),
    )

    assert (
        "localized UI/API draft introduced backend contract or session-state changes not present in requirements; "
        "do not invent new response fields or backend status plumbing unless the requirements explicitly require a payload/response change"
    ) in failures


def test_validate_draft_flags_export_state_ownership_ambiguity_for_localized_ui_requests(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        core_modules=[
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/components/PlanPanel.tsx",
        ],
        relevant_modules=[
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/components/PlanPanel.tsx",
            "src/prscope/web/frontend/src/lib/api.ts",
        ],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="localized export wiring in planning page and panel",
        risks=[],
        file_contents={
            "src/prscope/web/frontend/src/pages/PlanningView.tsx": "export function PlanningView() {}",
            "src/prscope/web/frontend/src/components/PlanPanel.tsx": "export function PlanPanel() {}",
            "src/prscope/web/frontend/src/lib/api.ts": "export function exportSession() {}\nexport function downloadFile() {}",
        },
        from_mental_model=False,
    )

    failures = agent.validate_draft(
        plan_content=(
            "# Plan\n\n## Summary\nShow export result.\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n"
            "## Changes\n- Update `PlanningView.tsx` to pass the necessary props to `PlanPanel`.\n\n"
            "## Files Changed\n- `src/prscope/web/frontend/src/pages/PlanningView.tsx`\n"
            "- `src/prscope/web/frontend/src/components/PlanPanel.tsx`\n"
            "- `src/prscope/web/frontend/src/pages/PlanningView.test.ts`\n\n"
            "## Architecture\nThe `PlanPanel` component will be updated to keep its internal state for export status, "
            "reducing reliance on external props from `PlanningView`.\n\n"
            "## Open Questions\n- None.\n"
        ),
        repo_understanding=repo_understanding,
        draft_phase="planner",
        requirements_text=(
            "Update the Planning UI to show the last export result and disable the export action while an export "
            "is already in progress. Reuse the existing exportSession and downloadFile helpers, preserve current "
            "PlanPanel behavior, and keep the change localized. Do not introduce new hooks, contexts, shared state "
            "layers, or background polling."
        ),
    )

    assert (
        "localized UI/API draft leaves export-state ownership ambiguous between `PlanningView.tsx` and `PlanPanel.tsx`; "
        "choose one owner and keep Files Changed, Architecture, and Implementation Steps consistent about whether state is passed as props or managed inside PlanPanel"
    ) in failures


def test_validate_draft_flags_planpanel_ownership_shift_when_behavior_must_be_preserved(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        core_modules=[
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/components/PlanPanel.tsx",
        ],
        relevant_modules=[
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/components/PlanPanel.tsx",
            "src/prscope/web/frontend/src/lib/api.ts",
        ],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="localized export wiring in planning page and panel",
        risks=[],
        file_contents={
            "src/prscope/web/frontend/src/pages/PlanningView.tsx": "export function PlanningView() {}",
            "src/prscope/web/frontend/src/components/PlanPanel.tsx": "export function PlanPanel() {}",
            "src/prscope/web/frontend/src/lib/api.ts": "export function exportSession() {}\nexport function downloadFile() {}",
        },
        from_mental_model=False,
    )

    failures = agent.validate_draft(
        plan_content=(
            "# Plan\n\n## Summary\nShow export result.\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n"
            "## Files Changed\n- `src/prscope/web/frontend/src/pages/PlanningView.tsx`\n"
            "- `src/prscope/web/frontend/src/components/PlanPanel.tsx`\n"
            "- `src/prscope/web/frontend/src/pages/PlanningView.test.ts`\n\n"
            "## Architecture\nThe `PlanPanel` will maintain its internal state for export status, thus reducing the dependency "
            "on `PlanningView`. It will now be responsible for displaying the last export result, enabling or disabling the "
            "export action, and managing error states internally.\n\n"
            "## Implementation Steps\n"
            "1. Update `src/prscope/web/frontend/src/pages/PlanningView.tsx`.\n"
            "2. Update `src/prscope/web/frontend/src/components/PlanPanel.tsx`.\n"
            "3. Add coverage in `src/prscope/web/frontend/src/pages/PlanningView.test.ts`.\n"
        ),
        repo_understanding=repo_understanding,
        draft_phase="planner",
        requirements_text=(
            "Update the Planning UI to show the last export result and disable the export action while an export "
            "is already in progress. Reuse the existing exportSession and downloadFile helpers, preserve current "
            "PlanPanel behavior, and keep the change localized. Do not introduce new hooks, contexts, shared state "
            "layers, or background polling."
        ),
    )

    assert (
        "localized UI/API draft shifts established export-state ownership into `PlanPanel.tsx` even though the request says to preserve current PlanPanel behavior; "
        "keep the existing `PlanningView.tsx`-owned wiring and limit PlanPanel changes to localized rendering/button behavior"
    ) in failures


def test_validate_draft_flags_speculative_result_format_questions_for_localized_ui_requests(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        core_modules=[
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/components/PlanPanel.tsx",
        ],
        relevant_modules=[
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/components/PlanPanel.tsx",
            "src/prscope/web/frontend/src/lib/api.ts",
        ],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="localized export wiring in planning page and panel",
        risks=[],
        file_contents={
            "src/prscope/web/frontend/src/pages/PlanningView.tsx": "export function PlanningView() {}",
            "src/prscope/web/frontend/src/components/PlanPanel.tsx": "export function PlanPanel() {}",
            "src/prscope/web/frontend/src/lib/api.ts": "export function exportSession() {}\nexport function downloadFile() {}",
        },
        from_mental_model=False,
    )

    failures = agent.validate_draft(
        plan_content=(
            "# Plan\n\n## Summary\nShow export result.\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n"
            "## Files Changed\n- `src/prscope/web/frontend/src/pages/PlanningView.tsx`\n"
            "- `src/prscope/web/frontend/src/components/PlanPanel.tsx`\n"
            "- `src/prscope/web/frontend/src/pages/PlanningView.test.ts`\n\n"
            "## Architecture\nKeep the change localized to the existing page/component wiring.\n\n"
            "## Open Questions\n"
            "- What specific format or details about the last export result should be shown in the UI?\n"
            "- Are there any additional requirements for logging the export actions or results?\n"
        ),
        repo_understanding=repo_understanding,
        draft_phase="planner",
        requirements_text=(
            "Update the Planning UI to show the last export result and disable the export action while an export "
            "is already in progress. Reuse the existing exportSession and downloadFile helpers, preserve current "
            "PlanPanel behavior, and keep the change localized. Do not introduce new hooks, contexts, shared state "
            "layers, or background polling."
        ),
    )

    assert (
        "localized UI/API draft turned a simple result-display request into speculative formatting open questions; "
        "default to a concise success/failure result presentation unless the requirements explicitly demand richer formatting"
    ) in failures
    assert (
        "localized UI/API draft introduced observability or telemetry scope not present in requirements; "
        "remove logging/telemetry/observability wording and keep the plan focused on existing UI wiring, "
        "API helper reuse, and tests"
    ) in failures


def test_validate_draft_flags_conditional_backend_contract_test_drift_for_localized_ui_requests(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/api.py"],
        core_modules=[
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/components/PlanPanel.tsx",
            "src/prscope/web/api.py",
        ],
        relevant_modules=[
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/components/PlanPanel.tsx",
            "src/prscope/web/frontend/src/lib/api.ts",
            "src/prscope/web/api.py",
        ],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts", "tests/test_web_api_models.py"],
        architecture_summary="frontend export wiring with existing API helpers",
        risks=[],
        file_contents={
            "src/prscope/web/frontend/src/pages/PlanningView.tsx": "export function PlanningView() {}",
            "src/prscope/web/frontend/src/components/PlanPanel.tsx": "export function PlanPanel() {}",
            "src/prscope/web/frontend/src/lib/api.ts": "export function exportSession() {}\nexport function downloadFile() {}",
            "src/prscope/web/api.py": "def get_session():\n    return {}\n",
        },
        from_mental_model=False,
    )

    result = agent.validate_draft_result(
        plan_content=(
            "# Plan\n\n## Summary\nShow export result.\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n"
            "## Changes\n- Keep the UI wiring localized and only adjust backend behavior if required.\n\n"
            "## Files Changed\n- `src/prscope/web/frontend/src/pages/PlanningView.tsx`\n"
            "- `src/prscope/web/frontend/src/components/PlanPanel.tsx`\n"
            "- `src/prscope/web/frontend/src/pages/PlanningView.test.ts`\n"
            "- `tests/test_web_api_models.py`\n\n"
            "## Architecture\nKeep the change localized to the existing component and verified helper wiring.\n\n"
            "## Implementation Steps\n1. Update the existing localized UI/component flow.\n"
            "2. Reuse `exportSession` and `downloadFile` exactly as already wired.\n\n"
            "## Test Strategy\n- Assert the requested user-visible behavior in the focused frontend regression test.\n"
            "- If the response shape changes, assert the localized backend contract through the existing API model regression test.\n\n"
            "## Rollback Plan\n- If the localized export UI behavior regresses, restore the prior UI flow.\n"
        ),
        repo_understanding=repo_understanding,
        draft_phase="refiner",
        requirements_text=(
            "Update the Planning UI to show the last export result and disable the export action while an export "
            "is already in progress. Reuse the existing exportSession and downloadFile helpers, preserve current "
            "PlanPanel behavior, and keep the change localized. Do not introduce new hooks, contexts, shared state "
            "layers, or background polling. Include focused frontend tests and any minimal backend contract test only "
            "if the response shape must change."
        ),
    )

    assert not result.ok
    assert "localized_scope_drift" in result.reason_codes
    assert "missing_localized_backend_grounding" not in result.reason_codes


def test_validate_draft_result_normalizes_reason_codes_and_retryability(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/api.py"],
        core_modules=["src/prscope/web/api.py"],
        relevant_modules=["src/prscope/web/api.py"],
        relevant_tests=["tests/test_web_api_models.py"],
        architecture_summary="single FastAPI module",
        risks=[],
        file_contents={
            "src/prscope/web/api.py": '@app.get("/health")\nasync def health():\n    return {"status": "ok"}\n'
        },
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


def test_validate_refinement_result_flags_under_scoped_localized_refinement(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        core_modules=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        relevant_modules=[
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/components/PlanPanel.tsx",
            "src/prscope/web/frontend/src/lib/api.ts",
        ],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="localized export status UI",
        risks=[],
        file_contents={
            "src/prscope/web/frontend/src/pages/PlanningView.tsx": "export function PlanningView() {}",
            "src/prscope/web/frontend/src/components/PlanPanel.tsx": "export function PlanPanel() {}",
            "src/prscope/web/frontend/src/lib/api.ts": "export function exportSession() {}\nexport function downloadFile() {}",
        },
        from_mental_model=False,
    )

    result = agent.validate_refinement_result(
        plan_content=(
            "# Plan\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Files Changed\n"
            "- `src/prscope/web/frontend/src/components/PlanPanel.tsx`\n\n"
            "## Architecture\nKeep the change localized.\n\n"
            "## Implementation Steps\n"
            "1. Update the existing localized UI/component flow.\n"
            "2. Reuse `exportSession` and `downloadFile` exactly as already wired.\n\n"
            "## Test Strategy\n"
            "- Assert the requested user-visible behavior in `src/prscope/web/frontend/src/pages/PlanningView.test.ts`.\n\n"
            "## Rollback Plan\n"
            "- If the localized export UI behavior regresses, restore the prior UI flow.\n"
        ),
        repo_understanding=repo_understanding,
        requirements_text=(
            "Update the Planning UI to show the last export result and disable the export action while an export "
            "is already in progress. Reuse the existing exportSession and downloadFile helpers, preserve current "
            "PlanPanel behavior, and keep the change localized. Include focused frontend tests."
        ),
    )

    assert not result.ok
    assert "missing_sections" in result.reason_codes
    assert "grounding_failure" in result.reason_codes
    assert "under-scoped draft: one file in Files Changed but multiple referenced files" in result.failure_messages
    assert (
        "Files Changed entries missing from Implementation Steps: src/prscope/web/frontend/src/components/PlanPanel.tsx"
    ) in result.failure_messages


def test_strip_localized_scope_drift_lines_removes_conditional_backend_contract_hedges() -> None:
    repaired = AuthorPlannerPipeline._strip_localized_scope_drift_lines(
        (
            "# Plan\n\n"
            "## Goals\n"
            "- Display the last export result.\n"
            "- Add `is_exporting` and `last_export_result` to `getSession`.\n\n"
            "## Non-Goals\n"
            "- Avoid broad backend changes beyond the `getSession` payload.\n\n"
            "## Files Changed\n"
            "- `src/prscope/web/frontend/src/pages/PlanningView.tsx`\n"
            "- `tests/test_web_api_models.py`\n\n"
            "## Architecture\n"
            "Keep the change localized to the existing component wiring.\n"
            "Only touch `src/prscope/web/api.py` if the response shape must change.\n\n"
            "## Test Strategy\n"
            "- Assert the requested user-visible behavior in the focused frontend regression test.\n"
            "- If the response shape changes, assert the localized backend contract through the existing API model regression test.\n"
        ),
        (
            "Update the Planning UI to show the last export result and disable the export action while an export is "
            "already in progress. Reuse the existing exportSession and downloadFile helpers, preserve current PlanPanel "
            "behavior, and keep the change localized. Include focused frontend tests and any minimal backend contract "
            "test only if the response shape must change."
        ),
    )

    assert "tests/test_web_api_models.py" not in repaired
    assert "src/prscope/web/api.py" not in repaired
    assert "backend contract through the existing API model regression test" not in repaired
    assert "is_exporting" not in repaired
    assert "getSession" not in repaired


def test_repair_sanitizer_strips_conditional_backend_contract_updates() -> None:
    current_plan = PlanDocument(
        title="Plan",
        summary="Keep it localized.",
        goals="- x",
        non_goals="- y",
        files_changed="- `src/prscope/web/frontend/src/pages/PlanningView.tsx`\n- `src/prscope/web/frontend/src/components/PlanPanel.tsx`",
        architecture="Keep the change localized to the existing UI flow and verified helper wiring.",
        implementation_steps="1. Update the component.\n2. Reuse existing helpers.",
        test_strategy="- Assert the requested user-visible behavior in the focused frontend regression test.",
        rollback_plan="- Restore the prior UI flow if the behavior regresses.",
        open_questions="- None.",
    )

    requirements = (
        "Update the Planning UI to show the last export result and disable the export action while an export is already "
        "in progress. Reuse the existing exportSession and downloadFile helpers, preserve current PlanPanel behavior, "
        "and keep the change localized. Include focused frontend tests and any minimal backend contract test only if "
        "the response shape must change."
    )

    sanitized_files = AuthorRepairService._sanitize_localized_ui_update(
        section_id="files_changed",
        content=(
            "- `src/prscope/web/frontend/src/pages/PlanningView.tsx`\n"
            "- `src/prscope/web/api.py`\n"
            "- `tests/test_web_api_models.py`\n"
        ),
        requirements=requirements,
        current_plan=current_plan,
    )
    sanitized_tests = AuthorRepairService._sanitize_localized_ui_update(
        section_id="test_strategy",
        content=(
            "- Assert the requested user-visible behavior in the focused frontend regression test.\n"
            "- If the response shape changes, assert the localized backend contract through the existing API model regression test.\n"
        ),
        requirements=requirements,
        current_plan=current_plan,
    )

    assert "src/prscope/web/api.py" not in sanitized_files
    assert "tests/test_web_api_models.py" not in sanitized_files
    assert "API model regression test" not in sanitized_tests


def test_repair_sanitizer_strips_conditional_backend_contract_goals() -> None:
    current_plan = PlanDocument(
        title="Plan",
        summary="Keep it localized.",
        goals="- Keep the change localized.\n- Reuse existing helpers.",
        non_goals="- Do not add backend contract changes.",
        files_changed="- `src/prscope/web/frontend/src/pages/PlanningView.tsx`",
        architecture="Keep the change localized to the existing UI flow and verified helper wiring.",
        implementation_steps="1. Update the component.\n2. Reuse existing helpers.",
        test_strategy="- Assert the requested user-visible behavior in the focused frontend regression test.",
        rollback_plan="- Restore the prior UI flow if the behavior regresses.",
        open_questions="- None.",
    )
    requirements = (
        "Update the Planning UI to show the last export result and disable the export action while an export is already "
        "in progress. Reuse the existing exportSession and downloadFile helpers, preserve current PlanPanel behavior, "
        "and keep the change localized. Include focused frontend tests and any minimal backend contract test only if "
        "the response shape must change."
    )

    sanitized_goals = AuthorRepairService._sanitize_localized_ui_update(
        section_id="goals",
        content=(
            "- Display the last export result.\n"
            "- Introduce minimal backend contract changes to `getSession` to expose `is_exporting` and `last_export_at`.\n"
        ),
        requirements=requirements,
        current_plan=current_plan,
    )
    sanitized_non_goals = AuthorRepairService._sanitize_localized_ui_update(
        section_id="non_goals",
        content="- Avoid broad backend changes beyond the `getSession` payload.",
        requirements=requirements,
        current_plan=current_plan,
    )

    assert "is_exporting" not in sanitized_goals
    assert "last_export_at" not in sanitized_goals
    assert "getSession" not in sanitized_non_goals


def test_author_planner_pipeline_deterministically_replaces_unverified_paths() -> None:
    pipeline = AuthorPlannerPipeline(
        tool_executor=SimpleNamespace(memory_block_callback=None, read_history={}),
        scan_repo_candidates=lambda **_: None,
        explore_repo=lambda **_: None,
        classify_complexity=lambda **_: "moderate",
        draft_plan=lambda **_: None,
        validate_draft=lambda **_: ValidationResult.success(),
    )
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        core_modules=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        relevant_modules=["src/prscope/web/frontend/src/components/PlanPanel.tsx"],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="localized export UI",
        risks=[],
        file_contents={},
        from_mental_model=False,
    )
    original_plan = (
        "# Plan\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Files Changed\n"
        "- `src/prscope/web/frontend/src/tests/PlanPanel.test.tsx`\n\n"
        "## Architecture\nKeep the change localized.\n\n"
        "## Implementation Steps\n1. Update the UI.\n\n"
        "## Test Strategy\n- Cover the export state in `src/prscope/web/frontend/src/tests/PlanPanel.test.tsx`.\n"
    )
    repaired_plan, repaired_result = pipeline._deterministic_plan_repairs(
        plan_content=original_plan,
        validation_result=ValidationResult(
            failure_messages=(
                "replace unverified path `src/prscope/web/frontend/src/tests/PlanPanel.test.tsx` with `src/prscope/web/frontend/src/pages/PlanningView.test.ts`",
            ),
            reason_codes=("unknown_file_reference",),
            retryable=True,
            failure_count=1,
        ),
        repo_understanding=repo_understanding,
        evidence_bundle=EvidenceBundle(
            relevant_files=("src/prscope/web/frontend/src/pages/PlanningView.tsx",),
            existing_components=(),
            test_targets=("src/prscope/web/frontend/src/pages/PlanningView.test.ts",),
            related_modules=(),
            existing_routes_or_helpers=(),
            evidence_notes=(),
        ),
        min_grounding_ratio=0.35,
        grounding_paths=set(),
        requirements="Keep the change localized and add focused frontend tests.",
    )

    assert repaired_result.ok
    assert "src/prscope/web/frontend/src/tests/PlanPanel.test.tsx" not in repaired_plan
    assert "src/prscope/web/frontend/src/pages/PlanningView.test.ts" in repaired_plan


def test_author_planner_pipeline_keeps_localized_frontend_owner_paths_in_files_changed() -> None:
    pipeline = AuthorPlannerPipeline(
        tool_executor=SimpleNamespace(memory_block_callback=None, read_history={}),
        scan_repo_candidates=lambda **_: None,
        explore_repo=lambda **_: None,
        classify_complexity=lambda **_: "moderate",
        draft_plan=lambda **_: None,
        validate_draft=lambda **_: ValidationResult.success(),
    )
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        core_modules=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        relevant_modules=[
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/components/PlanPanel.tsx",
            "src/prscope/web/frontend/src/lib/api.ts",
        ],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="localized export UI",
        risks=[],
        file_contents={},
        from_mental_model=False,
    )
    original_plan = (
        "# Plan\n\n## Summary\nKeep the change localized.\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Files Changed\n"
        "- `src/prscope/web/frontend/src/pages/PlanningView.test.ts`\n\n"
        "## Architecture\nKeep the change localized.\n\n"
        "## Implementation Steps\n1. Update the UI.\n\n"
        "## Test Strategy\n- Cover the export state in `src/prscope/web/frontend/src/pages/PlanningView.test.ts`.\n"
    )
    repaired_plan, repaired_result = pipeline._deterministic_plan_repairs(
        plan_content=original_plan,
        validation_result=ValidationResult(
            failure_messages=(
                "localized frontend UI change should reference a frontend regression target; mention `src/prscope/web/frontend/src/pages/PlanningView.test.ts`",
            ),
            reason_codes=("missing_tests",),
            retryable=True,
            failure_count=1,
        ),
        repo_understanding=repo_understanding,
        evidence_bundle=EvidenceBundle(
            relevant_files=(
                "src/prscope/web/frontend/src/pages/PlanningView.tsx",
                "src/prscope/web/frontend/src/components/PlanPanel.tsx",
                "src/prscope/web/frontend/src/lib/api.ts",
            ),
            existing_components=(),
            test_targets=("src/prscope/web/frontend/src/pages/PlanningView.test.ts",),
            related_modules=(),
            existing_routes_or_helpers=("exportSession", "downloadFile"),
            evidence_notes=(),
        ),
        min_grounding_ratio=0.35,
        grounding_paths=set(),
        requirements=(
            "Update the Planning UI to show the last export result and disable the export action while an export is already "
            "in progress. Reuse the existing exportSession and downloadFile helpers, preserve current PlanPanel behavior, "
            "and keep the change localized."
        ),
    )

    assert repaired_result.ok
    assert "src/prscope/web/frontend/src/pages/PlanningView.tsx" in repaired_plan
    assert "src/prscope/web/frontend/src/components/PlanPanel.tsx" in repaired_plan
    assert "src/prscope/web/frontend/src/lib/api.ts" in repaired_plan


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


def test_validate_draft_result_requires_frontend_regression_target_for_localized_ui_requests(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/api.py"],
        core_modules=[
            "src/prscope/web/api.py",
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
        ],
        relevant_modules=[
            "src/prscope/web/api.py",
            "src/prscope/web/frontend/src/components/PlanPanel.tsx",
            "src/prscope/web/frontend/src/lib/api.ts",
        ],
        relevant_tests=["tests/test_web_api_models.py", "src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="localized export status UI with small backend payload tweak",
        risks=[],
        file_contents={
            "src/prscope/web/frontend/src/lib/api.ts": "export function exportSession() {}\nexport function downloadFile() {}",
        },
        from_mental_model=False,
    )

    result = agent.validate_draft_result(
        plan_content=(
            "# Draft\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Files Changed\n"
            "- `src/prscope/web/frontend/src/components/PlanPanel.tsx`\n"
            "- `src/prscope/web/api.py`\n"
            "- `tests/test_web_api_models.py`\n\n"
            "## Architecture\nReuse `exportSession` and `downloadFile` while keeping any response shape tweak localized.\n\n"
            "## Open Questions\n- None.\n"
        ),
        repo_understanding=repo_understanding,
        draft_phase="planner",
        min_grounding_ratio=0.35,
        requirements_text=(
            "Update the Planning UI to show the last export result and disable the export action while an export is "
            "already in progress. Reuse the existing exportSession and downloadFile helpers, preserve current "
            "PlanPanel behavior, keep the change localized, and add focused frontend tests. Include a minimal backend "
            "contract test only if the response shape must change."
        ),
    )

    assert not result.ok
    assert "missing_tests" in result.reason_codes
    assert (
        "localized frontend UI change should reference a frontend regression target; mention "
        "`src/prscope/web/frontend/src/pages/PlanningView.test.ts`"
    ) in result.failure_messages


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
        "replace unverified path `src/prscope/web/lib/api.ts` with `src/prscope/web/frontend/src/lib/api.ts`" in failure
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


def test_validate_draft_result_ignores_pascal_case_ui_labels_when_requiring_helper_reuse(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        core_modules=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        relevant_modules=[
            "src/prscope/web/frontend/src/components/ActionBar.tsx",
            "src/prscope/web/frontend/src/lib/api.ts",
        ],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="frontend utilities menu flow",
        risks=[],
        file_contents={
            "src/prscope/web/frontend/src/components/ActionBar.tsx": "import { Download } from 'lucide-react';",
            "src/prscope/web/frontend/src/lib/api.ts": "export function downloadFile() {}\nexport function exportSession() {}",
        },
        from_mental_model=False,
    )

    result = agent.validate_draft_result(
        plan_content=(
            "# Draft\n\n## Summary\nx\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Files Changed\n"
            "- `src/prscope/web/frontend/src/components/ActionBar.tsx`\n"
            "- `src/prscope/web/frontend/src/pages/PlanningView.test.ts`\n\n"
            "## Architecture\nReuse the existing export helpers.\n\n## Open Questions\n- None.\n"
        ),
        repo_understanding=repo_understanding,
        draft_phase="planner",
        min_grounding_ratio=0.45,
        requirements_text=(
            "Add export actions to the ActionBar, reuse existing download/export helpers, and add tests."
        ),
    )

    assert "missing_helper_reuse" in result.reason_codes
    assert any("downloadFile" in failure for failure in result.failure_messages)
    assert not any("mention one of: Download" in failure for failure in result.failure_messages)


def test_validate_draft_result_reads_prioritized_api_helper_file_for_mixed_utilities_prompt(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        core_modules=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        relevant_modules=[
            "src/prscope/web/frontend/src/components/ActionBar.tsx",
            "src/prscope/web/frontend/src/components/PlanPanel.tsx",
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/components/IssueList.tsx",
            "src/prscope/web/frontend/src/components/ReviewNotes.tsx",
            "src/prscope/web/frontend/src/components/SessionBanner.tsx",
            "src/prscope/web/frontend/src/lib/api.ts",
        ],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="mixed utilities flow",
        risks=[],
        file_contents={
            "src/prscope/web/frontend/src/components/ActionBar.tsx": "export function ActionBar() {}",
            "src/prscope/web/frontend/src/components/PlanPanel.tsx": "export function PlanPanel() {}",
            "src/prscope/web/frontend/src/pages/PlanningView.tsx": "const snapshot = getSessionSnapshot(id);",
            "src/prscope/web/frontend/src/lib/api.ts": "export function getSessionSnapshot() {}",
        },
        from_mental_model=False,
    )
    agent.tool_executor.read_file = lambda path, max_lines=120: {
        "content": (
            "export function getSessionSnapshot() {}\n"
            "export function exportSession() {}\n"
            "export async function downloadFile() {}\n"
        )
        if path.endswith("/lib/api.ts")
        else ""
    }

    result = agent.validate_draft_result(
        plan_content=(
            "# Draft\n\n## Summary\nx\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Files Changed\n"
            "- `src/prscope/web/frontend/src/components/ActionBar.tsx`\n"
            "- `src/prscope/web/frontend/src/pages/PlanningView.test.ts`\n\n"
            "## Architecture\nReuse `getSessionSnapshot` without adding new routes.\n\n## Open Questions\n- None.\n"
        ),
        repo_understanding=repo_understanding,
        draft_phase="planner",
        min_grounding_ratio=0.45,
        requirements_text=(
            "Add a combined utilities menu that reuses existing export, download, and snapshot helpers, "
            "keeps PlanPanel behavior intact, and adds tests."
        ),
    )

    assert "missing_helper_reuse" in result.reason_codes
    assert any("exportSession" in failure for failure in result.failure_messages)
    assert any("downloadFile" in failure for failure in result.failure_messages)


def test_validate_draft_result_requires_existing_web_api_path_for_localized_backend_payload_tweak(
    tmp_path: Path,
) -> None:
    agent = _make_agent(tmp_path)
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/api.py"],
        core_modules=[
            "src/prscope/web/api.py",
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
        ],
        relevant_modules=[
            "src/prscope/web/api.py",
            "src/prscope/web/frontend/src/components/ActionBar.tsx",
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
        ],
        relevant_tests=["tests/test_web_api_models.py", "src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="localized frontend with small backend payload tweak",
        risks=[],
        file_contents={
            "src/prscope/web/api.py": "class SessionSnapshotResponse(BaseModel):\n    updated_at: str\n",
            "src/prscope/web/frontend/src/pages/PlanningView.tsx": "const snapshot = getSessionSnapshot(id);",
        },
        from_mental_model=False,
    )

    result = agent.validate_draft_result(
        plan_content=(
            "# Draft\n\n## Summary\nx\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Files Changed\n"
            "- `src/prscope/web/frontend/src/components/ActionBar.tsx`\n"
            "- `src/prscope/web/frontend/src/pages/PlanningView.test.ts`\n\n"
            "## Architecture\nIf the payload shape needs a small adjustment, keep the backend response change localized.\n\n"
            "## Open Questions\n- What payload fields need to be added for the summary chip?\n"
        ),
        repo_understanding=repo_understanding,
        draft_phase="planner",
        min_grounding_ratio=0.45,
        requirements_text=(
            "Add a compact summary chip in the ActionBar, reuse existing snapshot/download endpoints, and if a "
            "backend payload response shape tweak is needed keep it localized to the existing web API serialization path."
        ),
    )

    assert "missing_localized_backend_grounding" in result.reason_codes
    assert any("`src/prscope/web/api.py`" in failure for failure in result.failure_messages)
    assert any("`tests/test_web_api_models.py`" in failure for failure in result.failure_messages)


def test_validate_draft_result_prefers_frontend_snapshot_helper_name_over_backend_snake_case(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/api.py"],
        core_modules=[
            "src/prscope/web/api.py",
            "src/prscope/web/frontend/src/lib/api.ts",
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
        ],
        relevant_modules=[
            "src/prscope/web/api.py",
            "src/prscope/web/frontend/src/lib/api.ts",
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
        ],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="localized frontend snapshot reuse",
        risks=[],
        file_contents={
            "src/prscope/web/api.py": "async def get_session_snapshot():\n    return {}\n",
            "src/prscope/web/frontend/src/lib/api.ts": "export function getSessionSnapshot() {}\n",
            "src/prscope/web/frontend/src/pages/PlanningView.tsx": "const snapshot = getSessionSnapshot(id);\n",
        },
        from_mental_model=False,
    )

    result = agent.validate_draft_result(
        plan_content=(
            "# Draft\n\n## Summary\nx\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Files Changed\n"
            "- `src/prscope/web/frontend/src/components/ActionBar.tsx`\n"
            "- `src/prscope/web/frontend/src/pages/PlanningView.test.ts`\n\n"
            "## Architecture\nReuse `get_session_snapshot` from the existing helper layer.\n\n## Open Questions\n- None.\n"
        ),
        repo_understanding=repo_understanding,
        draft_phase="planner",
        min_grounding_ratio=0.45,
        requirements_text=(
            "Add a compact diagnostics summary in the ActionBar, reuse the existing frontend snapshot helper, "
            "and add tests."
        ),
    )

    assert "missing_helper_reuse" in result.reason_codes
    assert any("getSessionSnapshot" in failure for failure in result.failure_messages)
    assert not any("get_session_snapshot" in failure for failure in result.failure_messages)


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
        ValidationResult.success(),  # used by _deterministic_plan_repairs after repair
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
            failure_messages=(
                "missing test target reference; reference one of: src/prscope/web/frontend/src/pages/PlanningView.test.ts",
            ),
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
            failure_messages=(
                "missing test target reference; reference one of: src/prscope/web/frontend/src/pages/PlanningView.test.ts",
            ),
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
            failure_messages=(
                "missing test target reference; reference one of: src/prscope/web/frontend/src/pages/PlanningView.test.ts",
            ),
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
            failure_messages=(
                "missing test target reference; reference one of: src/prscope/web/frontend/src/pages/PlanningView.test.ts",
            ),
            reason_codes=("missing_tests",),
            retryable=True,
            failure_count=1,
        ),
        ValidationResult(
            failure_messages=(
                "missing test target reference; reference one of: src/prscope/web/frontend/src/pages/PlanningView.test.ts",
            ),
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

    agent = _make_agent(Path("."))
    architecture = agent._validation_service.extract_section(result.plan, "Architecture").lower()  # type: ignore[attr-defined]
    changes = agent._validation_service.extract_section(result.plan, "Changes").lower()  # type: ignore[attr-defined]
    assert "observability" not in architecture
    assert "logging" not in architecture
    assert "observability" not in changes
    assert "logging" not in changes
    assert result.draft_diagnostics["quality_gate_failures"] == []


@pytest.mark.asyncio
async def test_author_planner_pipeline_deterministically_repairs_frontend_state_scope_drift() -> None:
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/frontend/src/components/PlanPanel.tsx"],
        core_modules=["src/prscope/web/frontend/src/components/PlanPanel.tsx"],
        relevant_modules=[
            "src/prscope/web/frontend/src/components/PlanPanel.tsx",
            "src/prscope/web/frontend/src/lib/api.ts",
        ],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="frontend export wiring",
        risks=[],
        file_contents={
            "src/prscope/web/frontend/src/components/PlanPanel.tsx": "export function PlanPanel() {}",
            "src/prscope/web/frontend/src/lib/api.ts": "export function exportSession() {}\nexport function downloadFile() {}",
        },
        from_mental_model=False,
    )
    validations = [
        ValidationResult(
            failure_messages=(
                "localized UI/API draft introduced frontend state or polling abstractions not present in requirements; "
                "remove hooks/contexts/shared-state/state-management/polling wording and keep the plan focused on direct component wiring, "
                "existing helper reuse, PlanPanel compatibility, and tests",
            ),
            reason_codes=("localized_scope_drift",),
            retryable=True,
            failure_count=1,
        ),
        ValidationResult(
            failure_messages=(
                "localized UI/API draft introduced frontend state or polling abstractions not present in requirements; "
                "remove hooks/contexts/shared-state/state-management/polling wording and keep the plan focused on direct component wiring, "
                "existing helper reuse, PlanPanel compatibility, and tests",
            ),
            reason_codes=("localized_scope_drift",),
            retryable=True,
            failure_count=1,
        ),
        ValidationResult.success(),
    ]

    async def _draft_plan(**_: object) -> str:
        return (
            "# Draft\n\n## Summary\nShow export result.\n\n## Goals\n- x\n\n## Non-Goals\n"
            "- Do not introduce new hooks, contexts, shared state layers, or background polling.\n\n"
            "## Changes\n- Tighten export UI with polling safeguards, a `lastExportResult` state object, and component-local state.\n\n"
            "## Files Changed\n- `src/prscope/web/frontend/src/components/PlanPanel.tsx`\n"
            "- `src/prscope/web/frontend/src/lib/api.ts`\n"
            "- `src/prscope/web/frontend/src/pages/PlanningView.test.ts`\n\n"
            "## Architecture\nUse `useState` with an `isExporting` boolean, `lastExportResult`, local state, hooks/contexts, and background polling around `exportSession`.\n\n"
            "## Open Questions\n- None.\n"
        )

    pipeline = AuthorPlannerPipeline(
        tool_executor=SimpleNamespace(memory_block_callback=None, read_history={}),
        scan_repo_candidates=lambda **_: RepoCandidates(
            entrypoints=["src/prscope/web/frontend/src/components/PlanPanel.tsx"],
            source_modules=[
                "src/prscope/web/frontend/src/components/PlanPanel.tsx",
                "src/prscope/web/frontend/src/lib/api.ts",
            ],
            tests_and_config=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
            all_paths=[
                "src/prscope/web/frontend/src/components/PlanPanel.tsx",
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
        requirements=(
            "Update the Planning UI to show the last export result and disable the export action while an export "
            "is already in progress. Reuse the existing exportSession and downloadFile helpers, preserve current "
            "PlanPanel behavior, and keep the change localized. Do not introduce new hooks, contexts, shared state "
            "layers, or background polling."
        ),
        min_grounding_ratio=0.45,
        grounding_paths={"src/prscope/web/frontend/src/components/PlanPanel.tsx"},
        model_override=None,
        rejection_counts={},
        rejection_reasons=[],
        timeout_seconds_override=None,
    )

    agent = _make_agent(Path("."))
    architecture = agent._validation_service.extract_section(result.plan, "Architecture").lower()  # type: ignore[attr-defined]
    changes = agent._validation_service.extract_section(result.plan, "Changes").lower()  # type: ignore[attr-defined]
    assert "state management" not in architecture
    assert "hooks/contexts" not in architecture
    assert "background polling" not in architecture
    assert "usestate" not in architecture
    assert "isexporting" not in architecture
    assert "lastexportresult" not in architecture
    assert "local state" not in architecture
    assert "polling safeguards" not in changes
    assert "lastexportresult" not in changes
    assert "component-local state" not in changes
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
            failure_messages=(
                "missing test target reference; reference one of: src/prscope/web/frontend/src/pages/PlanningView.test.ts",
            ),
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
async def test_author_planner_pipeline_deterministically_repairs_missing_helper_reuse_reference() -> None:
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        core_modules=[
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/lib/api.ts",
        ],
        relevant_modules=[
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/components/ActionBar.tsx",
            "src/prscope/web/frontend/src/lib/api.ts",
        ],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="frontend utilities flow",
        risks=[],
        file_contents={
            "src/prscope/web/frontend/src/pages/PlanningView.tsx": "queryFn: () => getSessionSnapshot(id)",
            "src/prscope/web/frontend/src/components/ActionBar.tsx": "export function ActionBar() {}",
            "src/prscope/web/frontend/src/lib/api.ts": (
                "export function getSessionSnapshot() {}\n"
                "export function exportSession() {}\n"
                "export async function downloadFile() {}\n"
            ),
        },
        from_mental_model=False,
    )
    validations = [
        ValidationResult(
            failure_messages=(
                "missing explicit helper reuse reference for snapshot; mention one of: getSessionSnapshot",
            ),
            reason_codes=("missing_helper_reuse",),
            retryable=True,
            failure_count=1,
        ),
        ValidationResult(
            failure_messages=(
                "missing explicit helper reuse reference for snapshot; mention one of: getSessionSnapshot",
            ),
            reason_codes=("missing_helper_reuse",),
            retryable=True,
            failure_count=1,
        ),
        ValidationResult.success(),
    ]

    async def _draft_plan(**_: object) -> str:
        return (
            "# Draft\n\n## Summary\nx\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Changes\n- z\n\n"
            "## Files Changed\n- `src/prscope/web/frontend/src/components/ActionBar.tsx`\n"
            "- `src/prscope/web/frontend/src/pages/PlanningView.tsx`\n\n"
            "## Architecture\nReuse `exportSession` and `downloadFile` without adding new routes.\n\n"
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
        requirements="Add a combined utilities menu that reuses snapshot, export, and download helpers.",
        min_grounding_ratio=0.45,
        grounding_paths={"src/prscope/web/frontend/src/components/ActionBar.tsx"},
        model_override=None,
        rejection_counts={},
        rejection_reasons=[],
        timeout_seconds_override=None,
    )

    assert "getSessionSnapshot" in result.plan
    assert "src/prscope/web/frontend/src/lib/api.ts" in result.plan
    assert result.draft_diagnostics["quality_gate_failures"] == []


@pytest.mark.asyncio
async def test_author_planner_pipeline_deterministically_repairs_localized_backend_grounding() -> None:
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/api.py"],
        core_modules=[
            "src/prscope/web/api.py",
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
        ],
        relevant_modules=[
            "src/prscope/web/api.py",
            "src/prscope/web/frontend/src/components/ActionBar.tsx",
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
        ],
        relevant_tests=["tests/test_web_api_models.py", "src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="localized frontend with small backend payload tweak",
        risks=[],
        file_contents={
            "src/prscope/web/api.py": "class SessionSnapshotResponse(BaseModel):\n    updated_at: str\n",
            "src/prscope/web/frontend/src/pages/PlanningView.tsx": "const snapshot = getSessionSnapshot(id);",
        },
        from_mental_model=False,
    )
    validations = [
        ValidationResult(
            failure_messages=(
                "localized backend payload/response change must reference the existing API path; mention `src/prscope/web/api.py`",
                "localized backend payload/response change must reference the API model regression target; mention `tests/test_web_api_models.py`",
            ),
            reason_codes=("missing_localized_backend_grounding",),
            retryable=True,
            failure_count=2,
        ),
        ValidationResult(
            failure_messages=(
                "localized backend payload/response change must reference the existing API path; mention `src/prscope/web/api.py`",
                "localized backend payload/response change must reference the API model regression target; mention `tests/test_web_api_models.py`",
            ),
            reason_codes=("missing_localized_backend_grounding",),
            retryable=True,
            failure_count=2,
        ),
        ValidationResult.success(),
    ]

    async def _draft_plan(**_: object) -> str:
        return (
            "# Draft\n\n## Summary\nx\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Changes\n- z\n\n"
            "## Files Changed\n- `src/prscope/web/frontend/src/components/ActionBar.tsx`\n"
            "- `src/prscope/web/frontend/src/pages/PlanningView.tsx`\n"
            "- `src/prscope/web/frontend/src/pages/PlanningView.test.ts`\n\n"
            "## Architecture\nIf the payload shape needs a small adjustment, keep the backend response change localized.\n\n"
            "## Open Questions\n- What payload fields need to be added?\n"
        )

    pipeline = AuthorPlannerPipeline(
        tool_executor=SimpleNamespace(memory_block_callback=None, read_history={}),
        scan_repo_candidates=lambda **_: RepoCandidates(
            entrypoints=["src/prscope/web/api.py"],
            source_modules=[
                "src/prscope/web/api.py",
                "src/prscope/web/frontend/src/components/ActionBar.tsx",
                "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            ],
            tests_and_config=[
                "tests/test_web_api_models.py",
                "src/prscope/web/frontend/src/pages/PlanningView.test.ts",
            ],
            all_paths=[
                "src/prscope/web/api.py",
                "src/prscope/web/frontend/src/components/ActionBar.tsx",
                "src/prscope/web/frontend/src/pages/PlanningView.tsx",
                "tests/test_web_api_models.py",
                "src/prscope/web/frontend/src/pages/PlanningView.test.ts",
            ],
        ),
        explore_repo=lambda **_: repo_understanding,
        classify_complexity=lambda **_: "moderate",
        draft_plan=_draft_plan,
        validate_draft=lambda **_: validations.pop(0),
    )

    result = await pipeline.run(
        requirements=(
            "Add a compact summary chip in the ActionBar and, if a backend payload response shape tweak is needed, "
            "keep it localized to the existing web API serialization path."
        ),
        min_grounding_ratio=0.45,
        grounding_paths={"src/prscope/web/frontend/src/components/ActionBar.tsx"},
        model_override=None,
        rejection_counts={},
        rejection_reasons=[],
        timeout_seconds_override=None,
    )

    assert "src/prscope/web/api.py" in result.plan
    assert "tests/test_web_api_models.py" in result.plan
    assert result.draft_diagnostics["quality_gate_failures"] == []


@pytest.mark.asyncio
async def test_author_planner_pipeline_preserves_planpanel_file_when_requirements_anchor_compatibility() -> None:
    repo_understanding = RepoUnderstanding(
        entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
        core_modules=[
            "src/prscope/web/frontend/src/pages/PlanningView.tsx",
            "src/prscope/web/frontend/src/components/PlanPanel.tsx",
        ],
        relevant_modules=[
            "src/prscope/web/frontend/src/components/ActionBar.tsx",
            "src/prscope/web/frontend/src/components/PlanPanel.tsx",
            "src/prscope/web/frontend/src/lib/api.ts",
        ],
        relevant_tests=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
        architecture_summary="localized summary chip with PlanPanel compatibility",
        risks=[],
        file_contents={
            "src/prscope/web/frontend/src/components/ActionBar.tsx": "export function ActionBar() {}",
            "src/prscope/web/frontend/src/components/PlanPanel.tsx": "export function PlanPanel() {}",
            "src/prscope/web/frontend/src/lib/api.ts": "export function getSessionSnapshot() {}\nexport async function downloadFile() {}\n",
        },
        from_mental_model=False,
    )
    validations = [ValidationResult.success(), ValidationResult.success()]

    async def _draft_plan(**_: object) -> str:
        return (
            "# Draft\n\n## Summary\nx\n\n## Goals\n- x\n\n## Non-Goals\n- y\n\n## Changes\n- Keep PlanPanel behavior intact.\n\n"
            "## Files Changed\n- `src/prscope/web/frontend/src/components/ActionBar.tsx`\n"
            "- `src/prscope/web/frontend/src/lib/api.ts`\n\n"
            "## Architecture\nPlanPanel compatibility must remain intact while reusing existing helpers.\n\n"
            "## Open Questions\n- None.\n"
        )

    pipeline = AuthorPlannerPipeline(
        tool_executor=SimpleNamespace(memory_block_callback=None, read_history={}),
        scan_repo_candidates=lambda **_: RepoCandidates(
            entrypoints=["src/prscope/web/frontend/src/pages/PlanningView.tsx"],
            source_modules=[
                "src/prscope/web/frontend/src/components/ActionBar.tsx",
                "src/prscope/web/frontend/src/components/PlanPanel.tsx",
                "src/prscope/web/frontend/src/lib/api.ts",
            ],
            tests_and_config=["src/prscope/web/frontend/src/pages/PlanningView.test.ts"],
            all_paths=[
                "src/prscope/web/frontend/src/components/ActionBar.tsx",
                "src/prscope/web/frontend/src/components/PlanPanel.tsx",
                "src/prscope/web/frontend/src/lib/api.ts",
                "src/prscope/web/frontend/src/pages/PlanningView.test.ts",
            ],
        ),
        explore_repo=lambda **_: repo_understanding,
        classify_complexity=lambda **_: "simple",
        draft_plan=_draft_plan,
        validate_draft=lambda **_: validations.pop(0),
    )

    result = await pipeline.run(
        requirements="Add a compact summary chip to the ActionBar and keep current PlanPanel export and health behavior working during rollout.",
        min_grounding_ratio=0.45,
        grounding_paths={"src/prscope/web/frontend/src/components/ActionBar.tsx"},
        model_override=None,
        rejection_counts={},
        rejection_reasons=[],
        timeout_seconds_override=None,
    )

    assert "src/prscope/web/frontend/src/components/PlanPanel.tsx" in result.plan


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
        file_contents={
            "src/prscope/web/api.py": "@app.get('/health')\nasync def health():\n    return {'status': 'ok'}\n"
        },
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
        file_contents={
            "src/prscope/web/api.py": '@app.get("/health")\nasync def health():\n    return {"status": "healthy"}\n'
        },
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
    assert "do not prescribe hook APIs, state variable names, or explicit local-state object shapes" in user_prompt
    assert "`useState`, `useEffect`, `isExporting`, `lastExportResult`" in user_prompt
    assert "Do not reference planning runtime or discovery modules for frontend wiring work" in user_prompt
    assert "compatibility constraint or test target" in user_prompt
    assert "do not add observability, logging, telemetry, rollout controls, or platform notes" in str(
        messages[0]["content"]
    )
    assert "do not prescribe hook APIs, state variable names, or concrete local-state shapes" in str(
        messages[0]["content"]
    )


@pytest.mark.asyncio
async def test_author_repair_service_falls_back_after_google_json_contract_failures() -> None:
    calls: list[str | None] = []

    async def fake_llm_call(messages, **kwargs):  # type: ignore[no-untyped-def]
        del messages
        model_override = kwargs.get("model_override")
        calls.append(model_override)
        if model_override == "gemini-2.5-flash":
            response_text = '{"problem_understanding":"Need tighter scope"'
        else:
            response_text = json.dumps(
                {
                    "problem_understanding": "Need tighter scope",
                    "accepted_issues": ["Clarify fallback handling"],
                    "rejected_issues": [],
                    "root_causes": ["Model-specific JSON truncation"],
                    "repair_strategy": "Retry with a certified structured-output model.",
                    "target_sections": ["architecture"],
                    "revision_plan": "Update the architecture section with explicit fallback behavior.",
                }
            )
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response_text))])
        return response, str(model_override or "")

    repair_service = AuthorRepairService(fake_llm_call)
    repair = await repair_service.plan_repair(
        review=ReviewResult(
            strengths=[],
            architectural_concerns=[],
            risks=[],
            simplification_opportunities=[],
            blocking_issues=["Clarify fallback handling"],
            reviewer_questions=[],
            recommended_changes=[],
            design_quality_score=6.5,
            confidence="medium",
            review_complete=False,
            simplest_possible_design=None,
            primary_issue="Clarify fallback handling",
            resolved_issues=[],
            constraint_violations=[],
            issue_priority=["Clarify fallback handling"],
            prose="",
        ),
        plan=PlanDocument(
            title="Plan",
            summary="Summary",
            goals="Goals",
            non_goals="Non-goals",
            files_changed="`src/app.py`",
            architecture="Architecture",
            implementation_steps="1. Step",
            test_strategy="Tests",
            rollback_plan="Rollback",
            open_questions="- None.",
        ),
        requirements="Clarify provider-aware fallback handling.",
        model_override="gemini-2.5-flash",
        fallback_model_override="gpt-4o-mini",
    )

    assert repair.problem_understanding == "Need tighter scope"
    assert calls == ["gemini-2.5-flash", "gemini-2.5-flash", "gpt-4o-mini"]
