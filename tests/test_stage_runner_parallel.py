from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prscope.planning.runtime.author import StageRunner, _tool_calls_parallel_safe
from prscope.planning.runtime.tools import ToolExecutor


def _tc(name: str, args: dict) -> MagicMock:
    tc = MagicMock()
    tc.id = f"id-{name}-{args.get('path', args.get('pattern', 'x'))}"
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


def test_parallel_safe_requires_two_read_only_tools() -> None:
    assert _tool_calls_parallel_safe([_tc("read_file", {"path": "a"})]) is False
    assert _tool_calls_parallel_safe([_tc("read_file", {"path": "a"}), _tc("grep_code", {"pattern": "x"})])
    assert _tool_calls_parallel_safe([_tc("read_file", {"path": "a"}), _tc("get_memory_block", {"key": "k"})]) is False


@pytest.mark.asyncio
async def test_parallel_batch_runs_tools_concurrently(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    executor = ToolExecutor(tmp_path)
    emit = AsyncMock()

    async def llm_call(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("unused")

    runner = StageRunner(llm_call, executor, emit)
    calls = [
        _tc("read_file", {"path": "a.txt"}),
        _tc("read_file", {"path": "b.txt"}),
    ]
    orig = ToolExecutor.execute

    def slow_execute(self: ToolExecutor, raw_tool_call: object) -> dict:  # type: ignore[override]
        time.sleep(0.07)
        return orig(self, raw_tool_call)

    with patch.object(ToolExecutor, "execute", slow_execute):
        t0 = time.perf_counter()
        n, asked, _ts = await runner.execute_tool_calls(
            stage="test",
            conversation=[],
            content="",
            tool_calls=calls,
        )
        elapsed = time.perf_counter() - t0
    assert n == 2
    assert asked is False
    # Sequential would be ~0.14s+; parallel should be ~0.07s+.
    assert elapsed < 0.13


@pytest.mark.asyncio
async def test_mixed_batch_still_completes(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    executor = ToolExecutor(tmp_path)
    emit = AsyncMock()

    async def llm_call(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("unused")

    runner = StageRunner(llm_call, executor, emit)
    calls = [
        _tc("read_file", {"path": "a.txt"}),
        _tc("get_memory_block", {"key": "architecture"}),
    ]
    n, asked, _ts = await runner.execute_tool_calls(
        stage="test",
        conversation=[],
        content="",
        tool_calls=calls,
    )
    assert n == 2
    assert asked is False
