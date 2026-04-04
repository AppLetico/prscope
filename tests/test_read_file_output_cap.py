"""read_file_max_output_chars post-window truncation."""

from __future__ import annotations

from pathlib import Path

from prscope.config import PlanningToolsConfig
from prscope.planning.runtime.tools import ToolExecutor


def test_read_file_truncates_huge_line(tmp_path: Path) -> None:
    p = tmp_path / "wide.txt"
    p.write_text("x" * 5000, encoding="utf-8")
    cfg = PlanningToolsConfig(read_file_max_output_chars=200)
    ex = ToolExecutor(tmp_path, tools_config=cfg)
    out = ex.read_file(path="wide.txt", max_lines=5)
    assert out.get("content_truncated") is True
    assert len(out["content"]) <= 200
    assert "note" in out
