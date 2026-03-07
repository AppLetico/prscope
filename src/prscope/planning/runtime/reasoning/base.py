from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from .models import ReasoningContext, ReasoningDecision

DecisionT = TypeVar("DecisionT", bound=ReasoningDecision)


class Reasoner(ABC, Generic[DecisionT]):
    @abstractmethod
    async def decide(self, context: ReasoningContext) -> DecisionT:
        raise NotImplementedError
