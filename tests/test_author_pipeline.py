from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prscope.config import PlanningConfig
from prscope.planning.runtime.author import AuthorAgent, RepoUnderstanding
from prscope.planning.runtime.authoring.discovery import requirement_search_patterns
from prscope.planning.runtime.authoring.models import PlanDocument, RepairPlan, RepoCandidates
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
    async def _fake_llm_call(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        del args, kwargs
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
    )

    assert result.problem_understanding
    assert result.updates.get("open_questions") == "- None."
    assert "open_questions" in result.justification


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
        classify_complexity=lambda **_: "simple",
        draft_plan=_draft_plan,
        validate_draft=AuthorAgent(  # type: ignore[misc]
            config=PlanningConfig(author_model="gpt-4o-mini"),
            tool_executor=ToolExecutor(Path(".")),
        ).validate_draft,
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
        ).validate_draft,
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
