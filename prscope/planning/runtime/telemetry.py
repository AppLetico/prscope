"""
Shared telemetry helpers for planning runtime LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...pricing import CostEstimate, estimate_cost_usd


@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    model: str

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class CompletionTelemetry:
    usage: TokenUsage
    cost: CostEstimate


def extract_usage(response: Any, model: str) -> TokenUsage:
    usage_obj = getattr(response, "usage", None)
    prompt_tokens = int(
        getattr(usage_obj, "prompt_tokens", None)
        or getattr(usage_obj, "input_tokens", 0)
        or 0
    )
    completion_tokens = int(
        getattr(usage_obj, "completion_tokens", None)
        or getattr(usage_obj, "output_tokens", 0)
        or 0
    )
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model=model,
    )


def completion_telemetry(response: Any, model: str) -> CompletionTelemetry:
    usage = extract_usage(response, model)
    cost = estimate_cost_usd(model, usage.prompt_tokens, usage.completion_tokens)
    return CompletionTelemetry(usage=usage, cost=cost)

