"""
Token budget management utilities for planning runtime.
"""

from __future__ import annotations

from dataclasses import dataclass


class ContextWindowExceeded(RuntimeError):
    """Raised when minimum required context cannot fit in model window."""


def estimate_tokens(text: str) -> int:
    # Conservative generic heuristic across code/prose content.
    return max(1, int(len(text) / 3.5))


@dataclass
class TokenBudgetManager:
    context_window: int
    max_completion_tokens: int
    warn_ratio: float = 0.4
    enforce_ratio: float = 0.5

    @property
    def available_prompt_tokens(self) -> int:
        return max(0, self.context_window - self.max_completion_tokens)

    def enforce_required(self, required_blocks: list[str]) -> None:
        required = sum(estimate_tokens(block) for block in required_blocks)
        if required > self.available_prompt_tokens:
            raise ContextWindowExceeded(
                f"Required context exceeds model window: required={required}, "
                f"available={self.available_prompt_tokens}"
            )

    def injected_tokens(self, blocks: list[str]) -> int:
        return sum(estimate_tokens(block) for block in blocks if block)

    def injection_ratio(self, blocks: list[str]) -> float:
        if self.context_window <= 0:
            return 0.0
        return self.injected_tokens(blocks) / float(self.context_window)

    def allocate(self, text: str, remaining_prompt_tokens: int) -> tuple[str, int]:
        if remaining_prompt_tokens <= 0:
            return "", 0
        estimated = estimate_tokens(text)
        if estimated <= remaining_prompt_tokens:
            return text, estimated

        target_chars = int(remaining_prompt_tokens * 3.5)
        if target_chars <= 0:
            return "", 0
        marker = "\n[Memory truncated to fit budget]\n"
        head = max(1, target_chars - len(marker))
        trimmed = text[:head] + marker
        return trimmed, estimate_tokens(trimmed)

