"""Per-tool tool_result_max_chars and read_file output caps."""

from __future__ import annotations

from pathlib import Path

from prscope.config import PlanningToolsConfig
from prscope.planning.runtime.tools import ToolExecutor


def test_effective_tool_result_max_chars_per_tool(tmp_path: Path) -> None:
    cfg = PlanningToolsConfig(
        tool_result_max_chars=8000,
        tool_result_max_chars_by_tool={"grep_code": 1200},
    )
    ex = ToolExecutor(tmp_path, tools_config=cfg)
    assert ex._effective_tool_result_max_chars("read_file") == 8000
    assert ex._effective_tool_result_max_chars("grep_code") == 1200


def test_format_result_payload_offloads_at_per_tool_threshold(tmp_path: Path) -> None:
    cfg = PlanningToolsConfig(
        tool_result_max_chars=50_000,
        tool_result_max_chars_by_tool={"grep_code": 2000},
    )
    ex = ToolExecutor(tmp_path, tools_config=cfg)
    # JSON > 2000 chars but < 50k — should offload because of grep_code override
    big = {"pattern": "x", "results": [{"path": "a.py", "line": 1, "text": "y" * 3000}], "count": 1}
    out = ex._format_result_payload("c1", "grep_code", big)
    assert "stored_at" in out
    assert out.get("truncated") is True
