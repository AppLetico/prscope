"""
Structured event emission helper for planning runtime.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable


class AnalyticsEmitter:
    def __init__(self, callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None):
        self.callback = callback

    async def emit(self, event: dict[str, Any]) -> None:
        if self.callback is None:
            return
        maybe = self.callback(event)
        if asyncio.iscoroutine(maybe):
            await maybe

