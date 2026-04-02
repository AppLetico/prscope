"""Tests for planning.tools.path_allowlist (optional per-tool path prefixes)."""

from __future__ import annotations

from pathlib import Path

import pytest

from prscope.config import PlanningToolsConfig
from prscope.planning.runtime.tools import ToolExecutor, ToolSafetyError


def test_path_allowlist_allows_matching_prefix(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "x.txt").write_text("y", encoding="utf-8")

    cfg = PlanningToolsConfig(
        path_allowlist={
            "read_file": ["src"],
            "list_files": ["src", "."],
            "grep_code": ["src"],
            "glob_files": ["src"],
        }
    )
    ex = ToolExecutor(tmp_path, tools_config=cfg)

    assert ex.read_file(path="src/a.py")["path"] == "src/a.py"
    ex.list_files(path="src")
    ex.list_files(path=".")
    ex.grep_code(pattern="x", path="src")
    ex.glob_files(pattern="*.py", path="src")

    with pytest.raises(ToolSafetyError, match="path_allowlist"):
        ex.read_file(path="secret/x.txt")


def test_path_allowlist_empty_list_denies_tool(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("z", encoding="utf-8")
    cfg = PlanningToolsConfig(path_allowlist={"read_file": []})
    ex = ToolExecutor(tmp_path, tools_config=cfg)
    with pytest.raises(ToolSafetyError, match="disabled"):
        ex.read_file(path="a.txt")


def test_no_allowlist_unchanged(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("z", encoding="utf-8")
    ex = ToolExecutor(tmp_path, tools_config=PlanningToolsConfig())
    assert ex.read_file(path="b.txt")["path"] == "b.txt"
