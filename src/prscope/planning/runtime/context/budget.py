"""
Token budget management utilities for planning runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_tiktoken_encoder_cache: dict[str, Any] = {}


def _load_tiktoken_encoder(model_id: str) -> Any | None:
    """Best-effort OpenAI-style encoding for budgeting; None if unavailable."""
    key = model_id.strip().lower()
    if key in _tiktoken_encoder_cache:
        return _tiktoken_encoder_cache[key]
    try:
        import tiktoken  # type: ignore[import-untyped]
    except ImportError:
        _tiktoken_encoder_cache[key] = None
        return None
    # Map common Prscope ids to tiktoken model names
    if key.startswith("gpt-5") or key.startswith("gpt-4.1") or key.startswith("gpt-4o"):
        enc = tiktoken.encoding_for_model("gpt-4o")
    elif key.startswith("gpt-4") or key.startswith("gpt-3.5"):
        enc = tiktoken.encoding_for_model("gpt-4")
    else:
        try:
            enc = tiktoken.encoding_for_model(model_id)
        except Exception:  # noqa: BLE001
            enc = tiktoken.get_encoding("cl100k_base")
    _tiktoken_encoder_cache[key] = enc
    return enc


def estimate_tokens(
    text: str,
    *,
    model_id: str | None = None,
    estimator: str = "heuristic",
) -> int:
    """
    Token estimate for prompt budgeting.

    - heuristic: len/3.5 (fast, conservative for mixed prose/code)
    - tiktoken: uses tiktoken when installed and model looks OpenAI-compatible; else heuristic
    """
    if not text:
        return 0
    est = estimator.strip().lower() or "heuristic"
    if est == "tiktoken" and model_id:
        mid = model_id.strip().lower()
        if mid.startswith("gpt") or mid.startswith("o1") or mid.startswith("o3") or mid.startswith("o4"):
            enc = _load_tiktoken_encoder(model_id)
            if enc is not None:
                return max(1, len(enc.encode(text)))
    return max(1, int(len(text) / 3.5))


class ContextWindowExceeded(RuntimeError):
    """Raised when minimum required context cannot fit in model window."""


@dataclass
class TokenBudgetManager:
    context_window: int
    max_completion_tokens: int
    warn_ratio: float = 0.4
    enforce_ratio: float = 0.5
    reserved_prompt_tokens: int = 0
    model_id: str | None = None
    token_estimator: str = "heuristic"

    def _et(self, text: str) -> int:
        return estimate_tokens(text, model_id=self.model_id, estimator=self.token_estimator)

    @property
    def available_prompt_tokens(self) -> int:
        return max(
            0,
            self.context_window - self.max_completion_tokens - max(0, self.reserved_prompt_tokens),
        )

    def enforce_required(self, required_blocks: list[str]) -> None:
        required = sum(self._et(block) for block in required_blocks)
        if required > self.available_prompt_tokens:
            raise ContextWindowExceeded(
                f"Required context exceeds model window: required={required}, available={self.available_prompt_tokens}"
            )

    def injected_tokens(self, blocks: list[str]) -> int:
        return sum(self._et(block) for block in blocks if block)

    def injection_ratio(self, blocks: list[str]) -> float:
        if self.context_window <= 0:
            return 0.0
        return self.injected_tokens(blocks) / float(self.context_window)

    def allocate(self, text: str, remaining_prompt_tokens: int) -> tuple[str, int]:
        if remaining_prompt_tokens <= 0:
            return "", 0
        estimated = self._et(text)
        if estimated <= remaining_prompt_tokens:
            return text, estimated

        target_chars = int(remaining_prompt_tokens * 3.5)
        if target_chars <= 0:
            return "", 0
        marker = "\n[Memory truncated to fit budget]\n"
        head = max(1, target_chars - len(marker))
        trimmed = text[:head] + marker
        return trimmed, self._et(trimmed)
