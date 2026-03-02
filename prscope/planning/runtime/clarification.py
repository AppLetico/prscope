"""
Clarification gate primitives for pause/resume behavior.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


class ClarificationAborted(RuntimeError):
    """Raised when a clarification flow is manually aborted."""


@dataclass
class ClarificationRequest:
    question: str
    context: str = ""
    source: str = "author"


class ClarificationGate:
    def __init__(self, timeout_seconds: int = 600):
        self._event = asyncio.Event()
        self._answers: list[str] = []
        self._timeout_seconds = timeout_seconds
        self._aborted = False

    async def wait_for_answer(self) -> tuple[list[str], bool]:
        if self._aborted:
            raise ClarificationAborted("Clarification gate aborted")
        try:
            await asyncio.wait_for(self._event.wait(), timeout=self._timeout_seconds)
            if self._aborted:
                raise ClarificationAborted("Clarification gate aborted")
            return self._answers, False
        except asyncio.TimeoutError:
            return [], True

    def provide_answer(self, answers: list[str]) -> None:
        self._answers = [str(answer) for answer in answers]
        self._event.set()

    def abort(self) -> None:
        self._aborted = True
        self._event.set()

