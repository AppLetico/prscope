"""
Periodic SSE "thinking" pings while a long async phase runs (UX only).

Does not interrupt blocking work inside asyncio.to_thread (e.g. raw LiteLLM calls);
see docs/agent-harness.md.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any, TypeVar

T = TypeVar("T")


async def run_with_elapsed_thinking(
    awaitable: Awaitable[T],
    *,
    emit: Callable[[dict[str, Any]], Awaitable[None]],
    first_after_s: float,
    interval_s: float,
    message: str,
) -> T:
    """Emit ``thinking`` events after ``first_after_s``, then every ``interval_s`` until done.

    If ``first_after_s`` <= 0, the awaitable runs with no background task.
    If ``interval_s`` <= 0, only one ping fires after the initial delay.
    """
    if first_after_s <= 0:
        return await awaitable

    async def _pinger() -> None:
        await asyncio.sleep(first_after_s)
        while True:
            await emit({"type": "thinking", "message": message})
            if interval_s <= 0:
                break
            await asyncio.sleep(interval_s)

    ping = asyncio.create_task(_pinger())
    try:
        return await awaitable
    finally:
        ping.cancel()
        with suppress(asyncio.CancelledError):
            await ping
