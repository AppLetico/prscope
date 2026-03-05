from __future__ import annotations

import json
from pathlib import Path

from prscope.config import PlanningConfig
from prscope.planning.runtime.author import AuthorAgent, RepoUnderstanding
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
