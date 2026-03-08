"""
Model pricing and context window metadata for planning telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass

# Pricing last updated: 2026-03-08.
# Verify against provider docs before updating:
# - https://developers.openai.com/api/docs/pricing
# - https://www.anthropic.com/pricing
# - https://ai.google.dev/gemini-api/docs/pricing
#
# Values are USD per 1M tokens: (input, output). OpenAI = Standard tier; Gemini = <=200k tier.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5.1": (1.25, 10.00),
    "gpt-5.2": (1.75, 14.00),
    "gpt-5.3-codex": (1.75, 14.00),
    "gpt-5.4": (2.50, 15.00),
    "o1-mini": (1.10, 4.40),
    "o3": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-opus-4-20250514": (5.00, 25.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-opus-4-5": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-6": (5.00, 25.00),
    # Gemini: standard tier, text. Long-context (>200k) uses higher tier; we use <=200k here.
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-flash-lite": (0.075, 0.30),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash-lite-preview-09-2025": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-3.1-flash-lite-preview": (0.25, 1.50),
    "gemini-3.1-pro-preview": (2.00, 12.00),
}

MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 200_000,
    "gpt-4.1-mini": 200_000,
    "gpt-4.1-nano": 200_000,
    "gpt-5": 200_000,
    "gpt-5-mini": 200_000,
    "gpt-5-nano": 200_000,
    "gpt-5.1": 200_000,
    "gpt-5.2": 200_000,
    "gpt-5.3-codex": 200_000,
    "gpt-5.4": 1_050_000,
    "o1-mini": 200_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    "claude-3-haiku-20240307": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "claude-sonnet-4-20250514": 200_000,
    "claude-opus-4-20250514": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-opus-4-5": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-opus-4-6": 200_000,
    "gemini-2.0-flash": 1_048_576,
    "gemini-2.0-flash-lite": 1_048_576,
    "gemini-1.5-pro": 2_000_000,
    "gemini-2.5-flash-lite": 1_048_576,
    "gemini-2.5-flash-lite-preview-09-2025": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    "gemini-3-flash-preview": 1_048_576,
    "gemini-3.1-flash-lite-preview": 1_048_576,
    "gemini-3.1-pro-preview": 1_048_576,
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
