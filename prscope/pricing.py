"""
Model pricing and context window metadata for planning telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass

# Pricing last updated: 2026-03-01.
# Verify against provider docs before updating:
# - https://openai.com/pricing
# - https://www.anthropic.com/pricing
#
# Values are USD per 1M tokens: (input, output)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
}

MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "claude-3-5-sonnet-20241022": 200_000,
}


@dataclass
class CostEstimate:
    prompt_cost_usd: float
    completion_cost_usd: float
    total_cost_usd: float

    @property
    def display(self) -> str:
        return f"${self.total_cost_usd:.4f}"


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> CostEstimate:
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return CostEstimate(prompt_cost_usd=0.0, completion_cost_usd=0.0, total_cost_usd=0.0)
    input_per_million, output_per_million = pricing
    prompt_cost = (max(prompt_tokens, 0) / 1_000_000.0) * input_per_million
    completion_cost = (max(completion_tokens, 0) / 1_000_000.0) * output_per_million
    total = prompt_cost + completion_cost
    return CostEstimate(prompt_cost_usd=prompt_cost, completion_cost_usd=completion_cost, total_cost_usd=total)

