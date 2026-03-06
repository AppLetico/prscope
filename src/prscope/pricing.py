"""
Model pricing and context window metadata for planning telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass

# Pricing last updated: 2026-03-02.
# Verify against provider docs before updating:
# - https://openai.com/pricing
# - https://www.anthropic.com/pricing
# - https://cloud.google.com/vertex-ai/generative-ai/pricing
#
# Values are USD per 1M tokens: (input, output)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o3": (2.00, 8.00),
    "o4-mini": (1.10, 4.40),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-opus-4-20250514": (5.00, 25.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-opus-4-5": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-6": (5.00, 25.00),
    # Gemini pricing can vary by modality/long-context tier and preview status.
    # Use standard text-token rates here for unified telemetry.
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-3-flash": (0.50, 3.00),
    "gemini-3-pro": (2.00, 12.00),
    "gemini-3.1-pro": (2.00, 12.00),
}

MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "o3": 200_000,
    "o4-mini": 200_000,
    "claude-3-haiku-20240307": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-sonnet-4-20250514": 200_000,
    "claude-opus-4-20250514": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-opus-4-5": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-opus-4-6": 200_000,
    "gemini-2.0-flash": 1_048_576,
    "gemini-1.5-pro": 2_000_000,
    "gemini-2.5-flash-lite": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    "gemini-3-flash": 1_048_576,
    "gemini-3-pro": 1_048_576,
    "gemini-3.1-pro": 1_048_576,
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
