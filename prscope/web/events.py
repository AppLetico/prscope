"""
Session-scoped SSE event emitter.

This emitter uses per-subscriber queues so multiple tabs receive identical
streams for the same session.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any


class SessionEventEmitter:
    """In-memory event emitter with bounded per-session queues."""

    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}

    async def emit(self, session_id: str, event: dict[str, Any]) -> None:
        subscribers = self._subscribers.get(session_id)
        if not subscribers:
            return
        for queue in tuple(subscribers):
            await queue.put(event)

    async def subscribe(self, session_id: str) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self.maxsize)
        subscribers = self._subscribers.setdefault(session_id, set())
        subscribers.add(queue)
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            subscribers = self._subscribers.get(session_id)
            if subscribers is None:
                return
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(session_id, None)

    def cleanup(self, session_id: str) -> None:
        self._subscribers.pop(session_id, None)
