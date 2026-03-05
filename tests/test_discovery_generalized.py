from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from prscope.planning.runtime.discovery import (
    BOOTSTRAP_ROUTE_REGEX,
    CODE_SIGNALS,
    FRAMEWORKS,
    DiscoveryManager,
    Evidence,
    FeatureIntent,
    parse_questions,
)


def test_parse_questions_handles_plain_q_format() -> None:
    reply = """
Q1: What should the endpoint return?
A) A simple "OK"
B) JSON {"status":"ok"}
C) Include metadata
D) Other - describe preference
""".strip()
    questions = parse_questions(reply)
    assert len(questions) == 1
    assert questions[0].text == "What should the endpoint return?"
    assert questions[0].options[3].is_other is True


def test_parse_questions_handles_markdown_wrapped_blocks() -> None:
    reply = """
**Q2: Where should the endpoint be added?**
- A) In `app/api/routes.py`
- B) In a dedicated module
- C) In an existing router class
- D) Other — describe your preference
""".strip()
    questions = parse_questions(reply)
    assert len(questions) == 1
    assert questions[0].text == "Where should the endpoint be added?"


def test_extract_feature_intent_various_requests() -> None:
    manager = DiscoveryManager.__new__(DiscoveryManager)
    assert manager._extract_feature_intent("create health endpoint") is not None
    assert manager._extract_feature_intent("add authentication middleware") is not None
    assert manager._extract_feature_intent("implement rate limiting") is not None
    assert manager._extract_feature_intent("build websocket support") is not None
    assert manager._extract_feature_intent("create the new feature") is None


def test_should_bootstrap_scan_has_guard() -> None:
    manager = DiscoveryManager.__new__(DiscoveryManager)
    assert manager._should_bootstrap_scan("add documentation pages") is False
    assert manager._should_bootstrap_scan("build roadmap") is False
    assert manager._should_bootstrap_scan("add auth middleware") is True
    assert manager._should_bootstrap_scan("review api routes") is True


def test_route_file_score_penalizes_test_paths() -> None:
    manager = DiscoveryManager.__new__(DiscoveryManager)
    assert manager._route_file_score("backend/routes.py") > 0
    assert manager._route_file_score("src/server.ts") > 0
    assert manager._route_file_score("tests/test_routes.py") < manager._route_file_score("backend/routes.py")


def test_existing_feature_evidence_lines_prefers_runtime_references() -> None:
    manager = DiscoveryManager.__new__(DiscoveryManager)
    manager.bootstrap_insights_by_session = {
        "s1": {
            "existing_feature": True,
            "feature_label": "status endpoint",
            "feature_keywords": ["status", "endpoint"],
            "matched_paths": ["docs/ARCHITECTURE.md", "prscope/web/api.py"],
            "matched_evidence": [
                "`docs/ARCHITECTURE.md:4` Interface never imports from web layer.",
                "`README.md:12` API key configuration is documented.",
                "`prscope/web/api.py:111` @app.get('/status')",
            ],
        }
    }

    lines = manager._existing_feature_evidence_lines("s1")

    assert lines
    assert "prscope/web/api.py:111" in lines[0]
    assert all("ARCHITECTURE.md" not in line for line in lines)


def test_existing_feature_evidence_lines_fallback_prefers_runtime_path() -> None:
    manager = DiscoveryManager.__new__(DiscoveryManager)
    manager.bootstrap_insights_by_session = {
        "s1": {
            "existing_feature": True,
            "feature_label": "status endpoint",
            "feature_keywords": ["status", "endpoint"],
            "matched_paths": ["docs/README.md", "prscope/web/api.py"],
            "matched_evidence": [],
        }
    }

    lines = manager._existing_feature_evidence_lines("s1")

    assert lines == ["Found existing status endpoint in `prscope/web/api.py`"]


def test_select_scan_directories_skips_vendor_and_hidden() -> None:
    manager = DiscoveryManager.__new__(DiscoveryManager)
    ranked = manager._select_scan_directories(
        [
            {"path": "src", "type": "dir"},
            {"path": "src/server.ts", "type": "file"},
            {"path": "backend", "type": "dir"},
            {"path": "backend/routes.py", "type": "file"},
            {"path": "node_modules", "type": "dir"},
            {"path": ".github", "type": "dir"},
        ]
    )
    assert ranked
    assert "node_modules" not in ranked
    assert ".github" not in ranked


def test_build_signal_index_and_detect_code_signals() -> None:
    manager = DiscoveryManager.__new__(DiscoveryManager)
    matches = [
        {"path": "api/app.py", "line": 10, "text": "@app.get('/x')"},
        {"path": "api/middleware/auth.py", "line": 22, "text": "app.use(authMiddleware)"},
        {"path": "workers/tasks.py", "line": 18, "text": "@shared_task"},
        {"path": "cli/main.py", "line": 12, "text": "@click.command()"},
    ]
    index = manager._build_signal_index(matches)
    scores = manager._detect_code_signals(index)
    assert scores["route"] >= 1
    assert scores["middleware"] >= 1
    assert scores["worker"] >= 1
    assert scores["cli"] >= 1


def test_detect_framework_prefers_highest_score() -> None:
    manager = DiscoveryManager.__new__(DiscoveryManager)
    matches = [
        {"path": "src/server.ts", "line": 1, "text": "const app = express()"},
        {"path": "src/server.ts", "line": 2, "text": "app.get('/users', handler)"},
        {"path": "src/router.ts", "line": 5, "text": "router.post('/login', login)"},
        {"path": "api/main.py", "line": 2, "text": "from fastapi import FastAPI"},
    ]
    index = manager._build_signal_index(matches)
    assert manager._detect_framework(index) == "express"


def test_detect_architecture_from_signal_scores() -> None:
    manager = DiscoveryManager.__new__(DiscoveryManager)
    assert manager._detect_architecture({"route": 6, "middleware": 2}) == "api_service"
    assert manager._detect_architecture({"worker": 3, "cron": 1}) == "worker_service"
    assert manager._detect_architecture({"cli": 2}) == "cli_tool"
    assert manager._detect_architecture({}) is None


def test_location_score_prioritizes_runtime_code() -> None:
    manager = DiscoveryManager.__new__(DiscoveryManager)
    middleware = manager._location_score("src/middleware/auth.py")
    helper = manager._location_score("src/utils/auth_helpers.py")
    tests = manager._location_score("tests/test_auth.py")
    huge = manager._location_score("src/middleware/auth.py", file_line_count=10000)
    assert middleware > helper
    assert tests < middleware
    assert huge < middleware


def test_aggregate_evidence_keeps_highest_confidence_entry() -> None:
    manager = DiscoveryManager.__new__(DiscoveryManager)
    result = manager._aggregate_evidence(
        [
            Evidence(path="src/a.py", snippet="short", confidence=1),
            Evidence(path="src/a.py", snippet="longer", confidence=1),
            Evidence(path="src/a.py", snippet="best", confidence=5),
            Evidence(path="src/b.py", snippet="ok", confidence=2),
        ]
    )
    assert result[0].path == "src/a.py"
    assert result[0].confidence == 5
    assert len(result) == 2


@pytest.mark.asyncio
async def test_ingest_feature_evidence_marks_existing_feature() -> None:
    manager = DiscoveryManager.__new__(DiscoveryManager)
    manager.bootstrap_insights_by_session = {}

    async def _run_bootstrap_tool(*, tool_name, path=None, pattern=None, max_entries=120, max_results=80):
        del tool_name, path, pattern, max_entries, max_results
        return {"path": "src/server.ts", "content": "rate limiting middleware exists"}

    manager._run_bootstrap_tool = _run_bootstrap_tool
    intent = FeatureIntent(
        label="rate limiting",
        keywords=["rate", "limiting"],
        patterns=[r"\brate\b", r"\blimiting\b"],
    )
    payload = {"result": {"results": [{"path": "src/server.ts", "line": 44, "text": "app.get("}]}}
    output = await manager._ingest_feature_evidence_from_tool(
        session_id="s1",
        feature=intent,
        tool_name="grep_code",
        parsed_args={"path": "src/server.ts", "pattern": "rate"},
        tool_result_payload=payload,
    )
    assert output
    insight = manager.bootstrap_insights_by_session["s1"]
    assert insight["existing_feature"] is True
    assert insight["feature_label"] == "rate limiting"


@pytest.mark.asyncio
async def test_build_first_turn_bootstrap_context_resets_feature_cache() -> None:
    manager = DiscoveryManager.__new__(DiscoveryManager)
    manager.bootstrap_insights_by_session = {
        "s1": {"existing_feature": True, "feature_label": "health endpoint", "matched_paths": ["api/app.py"]}
    }
    manager.tool_executor = SimpleNamespace()

    async def _run_bootstrap_tool(*, tool_name, path=None, pattern=None, max_entries=120, max_results=80):
        del pattern, max_entries, max_results
        if tool_name == "list_files":
            return {"entries": [{"path": "src", "type": "dir"}, {"path": "src/server.ts", "type": "file"}]}
        if tool_name == "grep_code":
            return {"results": []}
        if tool_name == "read_file":
            return {"path": path, "content": ""}
        return None

    manager._run_bootstrap_tool = _run_bootstrap_tool
    context, _ = await manager._build_first_turn_bootstrap_context(
        "s1",
        [{"role": "user", "content": "add authentication middleware"}],
        1,
    )
    assert "Bootstrap Scan Evidence" in context
    assert manager.bootstrap_insights_by_session["s1"].get("feature_label") != "health endpoint"


def test_framework_registry_has_valid_patterns() -> None:
    assert FRAMEWORKS
    for framework in FRAMEWORKS:
        assert framework.route_patterns
        assert framework.file_patterns
        for pattern in framework.route_patterns:
            assert isinstance(pattern, re.Pattern)


def test_bootstrap_route_regex_is_compiled() -> None:
    assert isinstance(BOOTSTRAP_ROUTE_REGEX, re.Pattern)


def test_detect_code_signals_returns_positive_entries_only() -> None:
    manager = DiscoveryManager.__new__(DiscoveryManager)
    matches = [{"path": "src/server.ts", "line": 3, "text": "app.get('/x')"}]
    index = manager._build_signal_index(matches)
    scores = manager._detect_code_signals(index)
    assert scores["route"] >= 1
    for signal in CODE_SIGNALS:
        if signal.name != "route":
            assert signal.name not in scores


def test_discovery_engine_has_no_feature_specific_logic() -> None:
    source = Path("prscope/planning/runtime/discovery.py").read_text()
    tree = ast.parse(source)
    forbidden_tokens = ["health", "graphql", "websocket", "celery"]
    allow = {"_extract_feature_intent"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name not in allow:
            function_source = (ast.get_source_segment(source, node) or "").lower()
            for token in forbidden_tokens:
                assert token not in function_source, f"forbidden token '{token}' in {node.name}"
