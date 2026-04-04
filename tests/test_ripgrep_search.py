"""Tests for ripgrep-backed search and ToolExecutor integration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from prscope.config import PlanningToolsConfig
from prscope.planning.runtime.ripgrep_search import (
    grep_code_python,
    grep_code_ripgrep,
    resolve_grep_backend,
)
from prscope.planning.runtime.tools import IGNORED_DIRS, ToolExecutor, ToolSafetyError


def test_resolve_grep_backend() -> None:
    assert resolve_grep_backend("python", "rg") == "python"
    assert resolve_grep_backend("auto", "nonexistent_binary_xyz") == "python"


def test_grep_code_python_matches(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("hello = 1\ndef foo():\n    pass\n", encoding="utf-8")
    sub = tmp_path / "src"
    sub.mkdir()
    (sub / "b.py").write_text("# skip\n", encoding="utf-8")
    rows = grep_code_python(
        repo_root=tmp_path,
        search_root=tmp_path,
        pattern=r"def\s+foo",
        max_results=10,
        ignored_dir_parts=frozenset(IGNORED_DIRS),
        binary_suffixes=frozenset({".png"}),
        case_insensitive=False,
    )
    assert len(rows) == 1
    assert rows[0]["path"].replace("\\", "/") == "a.py"
    assert rows[0]["line"] == 2


def test_grep_code_python_case_insensitive(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("Hello World\n", encoding="utf-8")
    rows = grep_code_python(
        repo_root=tmp_path,
        search_root=tmp_path,
        pattern="hello",
        max_results=5,
        ignored_dir_parts=frozenset(IGNORED_DIRS),
        binary_suffixes=frozenset(),
        case_insensitive=True,
    )
    assert len(rows) == 1


def test_grep_code_ripgrep_parses_json(tmp_path: Path) -> None:
    repo = tmp_path
    (repo / "sample.py").write_text("x = 42\n", encoding="utf-8")

    fake_stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "match",
                    "data": {
                        "path": {"text": "sample.py"},
                        "lines": {"text": "x = 42\n"},
                        "line_number": 1,
                        "submatches": [],
                    },
                }
            )
        ]
    )

    with patch("prscope.planning.runtime.ripgrep_search.subprocess.run") as run_mock:
        run_mock.return_value = type("R", (), {"returncode": 0, "stdout": fake_stdout, "stderr": ""})()

        rows, err, truncated = grep_code_ripgrep(
            repo_root=repo,
            search_root=repo,
            pattern=r"x\s*=",
            max_results=5,
            command="rg",
            timeout_seconds=5.0,
            max_columns=500,
            respect_ignore_files=True,
            output_mode="content",
            glob=None,
            type_tag=None,
            case_insensitive=False,
        )
        assert err is None
        assert not truncated
        assert len(rows) == 1
        assert rows[0]["path"] == "sample.py"
        assert rows[0]["line"] == 1


def test_grep_code_ripgrep_offset_skips_rows(tmp_path: Path) -> None:
    repo = tmp_path
    lines = []
    for i in range(3):
        lines.append(
            json.dumps(
                {
                    "type": "match",
                    "data": {
                        "path": {"text": "a.py"},
                        "lines": {"text": f"line{i}\n"},
                        "line_number": i + 1,
                        "submatches": [],
                    },
                }
            )
        )
    fake_stdout = "\n".join(lines)

    with patch("prscope.planning.runtime.ripgrep_search.subprocess.run") as run_mock:
        run_mock.return_value = type("R", (), {"returncode": 0, "stdout": fake_stdout, "stderr": ""})()

        rows, err, truncated = grep_code_ripgrep(
            repo_root=repo,
            search_root=repo,
            pattern=r"line",
            max_results=10,
            command="rg",
            timeout_seconds=5.0,
            max_columns=500,
            respect_ignore_files=True,
            output_mode="content",
            glob=None,
            type_tag=None,
            case_insensitive=False,
            offset=1,
        )
        assert err is None
        assert not truncated
        assert len(rows) == 2
        assert rows[0]["text"] == "line1"


def test_tool_executor_glob_files(tmp_path: Path) -> None:
    (tmp_path / "one.py").write_text("a", encoding="utf-8")
    d = tmp_path / "src"
    d.mkdir()
    (d / "two.py").write_text("b", encoding="utf-8")
    ex = ToolExecutor(tmp_path, tools_config=PlanningToolsConfig(glob_max_results=100))
    out = ex.glob_files("**/*.py", path=".")
    paths = set(out["results"])
    assert "one.py" in paths
    assert "src/two.py" in paths


def test_tool_executor_glob_files_brace_pattern_includes_note(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("1", encoding="utf-8")
    ex = ToolExecutor(tmp_path)
    out = ex.glob_files("**/*.{py,ts}", path=".")
    assert "note" in out
    assert "brace" in out["note"].lower()


def test_tool_executor_grep_matches_python_fallback(tmp_path: Path) -> None:
    (tmp_path / "t.py").write_text("unique_marker_xyz = 1\n", encoding="utf-8")
    ex = ToolExecutor(tmp_path, tools_config=PlanningToolsConfig(grep_backend="python"))
    payload = ex.grep_code(r"unique_marker_xyz", path=".", max_results=10)
    assert payload["grep_backend"] == "python"
    assert payload["output_mode"] == "content"
    assert len(payload["results"]) == 1
    assert payload["results"][0]["line"] == 1


def test_tool_executor_grep_files_with_matches_python(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("z = 1\nz = 2\n", encoding="utf-8")
    ex = ToolExecutor(tmp_path, tools_config=PlanningToolsConfig(grep_backend="python"))
    payload = ex.grep_code("z", output_mode="files_with_matches", max_results=10)
    assert len(payload["results"]) == 1
    assert payload["results"][0]["path"] == "a.py"
    assert payload["results"][0]["line"] == 0


def test_tool_executor_execute_glob_files(tmp_path: Path) -> None:
    (tmp_path / "f.py").write_text("1", encoding="utf-8")
    ex = ToolExecutor(tmp_path)
    raw = {
        "id": "c1",
        "function": {"name": "glob_files", "arguments": json.dumps({"pattern": "*.py", "path": "."})},
    }
    out = ex.execute(raw)
    res = out["result"]["result"]
    assert "f.py" in res["results"]


def test_tool_executor_invalid_regex(tmp_path: Path) -> None:
    ex = ToolExecutor(tmp_path, tools_config=PlanningToolsConfig(grep_backend="python"))
    with pytest.raises(ToolSafetyError):
        ex.grep_code("(", path=".", max_results=5)
