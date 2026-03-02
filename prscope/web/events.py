"""
Session-scoped SSE event emitter.

This emitter intentionally uses a single queue per session (single-consumer
model). For this local tool, one active UI tab per session is supported.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any


class SessionEventEmitter:
    """In-memory event emitter with bounded per-session queues."""

    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self._queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._subscribers: dict[str, int] = {}

    def get_queue(self, session_id: str) -> asyncio.Queue[dict[str, Any]]:
        queue = self._queues.get(session_id)
        if queue is None:
            queue = asyncio.Queue(maxsize=self.maxsize)
            self._queues[session_id] = queue
        return queue

    def _subscriber_count(self, session_id: str) -> int:
        return self._subscribers.get(session_id, 0)

    async def emit(self, session_id: str, event: dict[str, Any]) -> None:
        # Avoid queue buildup when nobody is listening.
        if self._subscriber_count(session_id) == 0:
            return
        queue = self.get_queue(session_id)
        # Explicit drop-oldest strategy under pressure.
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await queue.put(event)

    async def subscribe(self, session_id: str) -> AsyncIterator[dict[str, Any]]:
        queue = self.get_queue(session_id)
        self._subscribers[session_id] = self._subscriber_count(session_id) + 1
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            remaining = max(0, self._subscriber_count(session_id) - 1)
            if remaining == 0:
                self._subscribers.pop(session_id, None)
            else:
                self._subscribers[session_id] = remaining

    def cleanup(self, session_id: str) -> None:
        self._queues.pop(session_id, None)
        self._subscribers.pop(session_id, None)
