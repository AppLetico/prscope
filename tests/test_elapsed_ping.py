from __future__ import annotations

import asyncio

import pytest

from prscope.planning.runtime.elapsed_ping import run_with_elapsed_thinking


@pytest.mark.asyncio
async def test_run_with_elapsed_thinking_skips_ping_when_disabled() -> None:
    emitted: list[dict] = []

    async def emit(event: dict) -> None:
        emitted.append(event)

    async def work() -> str:
        return "ok"

    out = await run_with_elapsed_thinking(
        work(),
        emit=emit,
        first_after_s=0.0,
        interval_s=35.0,
        message="ping",
    )
    assert out == "ok"
    assert emitted == []


@pytest.mark.asyncio
async def test_run_with_elapsed_thinking_completes_before_first_ping() -> None:
    emitted: list[dict] = []

    async def emit(event: dict) -> None:
        emitted.append(event)

    async def work() -> str:
        await asyncio.sleep(0)
        return "done"

    out = await run_with_elapsed_thinking(
        work(),
        emit=emit,
        first_after_s=10.0,
        interval_s=5.0,
        message="Still working...",
    )
    assert out == "done"
    assert emitted == []


@pytest.mark.asyncio
async def test_run_with_elapsed_thinking_emits_once_when_interval_zero() -> None:
    emitted: list[dict] = []

    async def emit(event: dict) -> None:
        emitted.append(event)

    started = asyncio.Event()

    async def work() -> str:
        await started.wait()
        return "end"

    task = asyncio.create_task(
        run_with_elapsed_thinking(
            work(),
            emit=emit,
            first_after_s=0.05,
            interval_s=0.0,
            message="ping",
        )
    )
    await asyncio.sleep(0.15)
    started.set()
    assert await task == "end"
    assert len(emitted) >= 1
    assert emitted[0]["type"] == "thinking"
    assert emitted[0]["message"] == "ping"
