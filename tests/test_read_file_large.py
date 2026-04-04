"""Tests for streaming vs fast-path read_file behavior (large on-disk files)."""

from __future__ import annotations

from pathlib import Path

import pytest

from prscope.config import PlanningToolsConfig
from prscope.planning.runtime.read_file_range import read_file_slice, read_file_slice_fast
from prscope.planning.runtime.tools import ToolExecutor, ToolSafetyError


def _omit_path(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "path"}


@pytest.fixture
def sample_text(tmp_path: Path) -> Path:
    p = tmp_path / "sample.txt"
    p.write_text(
        "line1\nline2\nline3\r\nline4\n\nafter_blank\n",
        encoding="utf-8",
    )
    return p


def test_read_file_slice_streaming_matches_fast_path_default(sample_text: Path) -> None:
    fast = read_file_slice_fast(sample_text, max_lines=3)
    stream = read_file_slice(
        sample_text,
        max_lines=3,
        fast_path_max_bytes=1,
    )
    assert fast == stream


def test_read_file_slice_streaming_matches_fast_path_start_line(sample_text: Path) -> None:
    fast = read_file_slice_fast(sample_text, max_lines=2, start_line=3)
    stream = read_file_slice(
        sample_text,
        max_lines=2,
        start_line=3,
        fast_path_max_bytes=1,
    )
    assert fast == stream


def test_read_file_slice_streaming_matches_fast_path_around_line(sample_text: Path) -> None:
    fast = read_file_slice_fast(sample_text, around_line=3, radius=1)
    stream = read_file_slice(
        sample_text,
        around_line=3,
        radius=1,
        fast_path_max_bytes=1,
    )
    assert fast == stream


def test_tool_executor_read_file_parity_with_streaming_forced(sample_text: Path) -> None:
    cfg = PlanningToolsConfig(read_file_fast_path_max_bytes=1)
    ex = ToolExecutor(sample_text.parent, tools_config=cfg)
    rel = "sample.txt"
    fast_ex = ToolExecutor(sample_text.parent, tools_config=PlanningToolsConfig())
    for kwargs in (
        {"max_lines": 3},
        {"max_lines": 2, "start_line": 3},
        {"around_line": 3, "radius": 1},
    ):
        a = _omit_path(fast_ex.read_file(path=rel, **kwargs))
        b = _omit_path(ex.read_file(path=rel, **kwargs))
        assert a == b, kwargs


def test_tool_executor_rejects_file_over_max_bytes(tmp_path: Path) -> None:
    p = tmp_path / "huge.txt"
    p.write_bytes(b"x" * 32)
    cfg = PlanningToolsConfig(read_file_fast_path_max_bytes=10_000, read_file_max_file_bytes=16)
    ex = ToolExecutor(tmp_path, tools_config=cfg)
    with pytest.raises(ToolSafetyError, match="File too large"):
        ex.read_file(path="huge.txt")
