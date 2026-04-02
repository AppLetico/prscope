"""Transient LLM failure detection and backoff for LiteLLM/OpenAI-style errors."""

from __future__ import annotations

import asyncio
import random
import re
import time
from typing import Any


def _http_status_from_exception(exc: BaseException) -> int | None:
    sc = getattr(exc, "status_code", None)
    if isinstance(sc, int):
        return sc
    response = getattr(exc, "response", None)
    rsc = getattr(response, "status_code", None) if response is not None else None
    if isinstance(rsc, int):
        return rsc
    return None


def extract_retry_after_seconds(exc: BaseException) -> float | None:
    """Parse Retry-After from a provider exception when present (seconds or HTTP-date)."""
    response = getattr(exc, "response", None)
    headers: Any = None
    if response is not None:
        headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:  # noqa: BLE001
        return None
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        sec = float(raw)
        return sec if sec > 0 else None
    text = str(raw).strip()
    if not text:
        return None
    try:
        sec = float(text)
        return sec if sec > 0 else None
    except ValueError:
        pass
    return None


_TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


def is_transient_llm_failure(exc: BaseException) -> bool:
    """Return True when the failure is plausibly transient (retry may help)."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    status = _http_status_from_exception(exc)
    if status is not None and status in _TRANSIENT_STATUS:
        return True
    name = type(exc).__name__.lower()
    if "timeout" in name or "ratelimit" in name.replace("_", "") or "throttl" in name:
        return True
    try:
        import litellm

        for attr in (
            "RateLimitError",
            "ServiceUnavailableError",
            "APIConnectionError",
            "InternalServerError",
        ):
            err_cls = getattr(litellm, attr, None)
            if err_cls is not None and isinstance(exc, err_cls):
                return True
    except Exception:  # noqa: BLE001
        pass
    try:
        from openai import APIError, APIStatusError, APITimeoutError, RateLimitError

        if isinstance(exc, (RateLimitError, APITimeoutError)):
            return True
        if isinstance(exc, APIStatusError):
            sc = getattr(exc, "status_code", None)
            if isinstance(sc, int) and sc in _TRANSIENT_STATUS:
                return True
        if isinstance(exc, APIError):
            sc = getattr(exc, "status_code", None)
            if isinstance(sc, int) and sc in _TRANSIENT_STATUS:
                return True
    except Exception:  # noqa: BLE001
        pass

    msg = str(exc).lower()
    if any(
        s in msg
        for s in (
            "rate limit",
            "too many requests",
            "429",
            "overloaded",
            "server_error",
            "503",
            "529",
            "temporarily unavailable",
            "try again",
            "timeout",
            "timed out",
            "connection error",
            "connection reset",
            "eof occurred",
        )
    ):
        return True
    return False


# Loose parse for "retry after X seconds" in provider messages (fallback when no header).
_RETRY_AFTER_MSG = re.compile(r"retry\s+after\s+(\d+)\s*s", re.IGNORECASE)


def extract_retry_after_from_message(exc: BaseException) -> float | None:
    m = _RETRY_AFTER_MSG.search(str(exc))
    if not m:
        return None
    try:
        sec = float(m.group(1))
        return sec if sec > 0 else None
    except ValueError:
        return None


def compute_retry_delay_seconds(
    *,
    attempt_index: int,
    initial_delay_seconds: float,
    max_delay_seconds: float,
    backoff_multiplier: float,
    respect_retry_after: bool,
    exc: BaseException | None,
) -> float:
    """Exponential backoff with jitter; honors Retry-After when configured."""
    base = max(0.0, float(initial_delay_seconds))
    mult = max(1.0, float(backoff_multiplier))
    cap = max(base, float(max_delay_seconds))
    exp = min(cap, base * (mult ** max(0, attempt_index)))
    jitter = random.uniform(0.0, min(1.0, exp * 0.1))
    delay = min(cap, exp + jitter)
    if respect_retry_after and exc is not None:
        ra = extract_retry_after_seconds(exc)
        if ra is None:
            ra = extract_retry_after_from_message(exc)
        if ra is not None:
            delay = min(cap, max(delay, ra))
    return delay


def sync_sleep(seconds: float) -> None:
    time.sleep(max(0.0, seconds))


async def async_sleep(seconds: float) -> None:
    await asyncio.sleep(max(0.0, seconds))


def is_non_chat_model_hint(exc: BaseException) -> bool:
    """Heuristic matching existing author/discovery fallback logic."""
    err_text = str(exc).lower()
    return (
        "not a chat model" in err_text
        or "v1/chat/completions" in err_text
        or "did you mean to use v1/completions" in err_text
    )
