"""Tests for planning.runtime.llm_retry helpers."""

from __future__ import annotations

import asyncio

from prscope.planning.runtime.llm_retry import (
    compute_retry_delay_seconds,
    extract_retry_after_from_message,
    extract_retry_after_seconds,
    is_non_chat_model_hint,
    is_transient_llm_failure,
)


class _ExcWithResponse:
    def __init__(self, status_code: int | None = None, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.response = _Resp(headers or {})


class _Resp:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers
        self.status_code = 429


def test_extract_retry_after_header() -> None:
    exc = _ExcWithResponse(headers={"retry-after": "12"})
    assert extract_retry_after_seconds(exc) == 12.0


def test_extract_retry_after_from_message() -> None:
    exc = RuntimeError("Please retry after 30s when capacity is available.")
    assert extract_retry_after_from_message(exc) == 30.0


def test_is_non_chat_model_hint() -> None:
    assert is_non_chat_model_hint(RuntimeError("model is not a chat model"))
    assert not is_non_chat_model_hint(RuntimeError("invalid api key"))


def test_is_transient_429() -> None:
    exc = _ExcWithResponse(status_code=429)
    assert is_transient_llm_failure(exc)


def test_is_transient_timeout() -> None:
    assert is_transient_llm_failure(asyncio.TimeoutError())


def test_compute_retry_delay_respects_retry_after() -> None:
    exc = _ExcWithResponse(headers={"retry-after": "5"})
    d = compute_retry_delay_seconds(
        attempt_index=0,
        initial_delay_seconds=0.1,
        max_delay_seconds=60.0,
        backoff_multiplier=2.0,
        respect_retry_after=True,
        exc=exc,
    )
    assert d >= 5.0


def test_compute_retry_delay_no_retry_after() -> None:
    d = compute_retry_delay_seconds(
        attempt_index=1,
        initial_delay_seconds=1.0,
        max_delay_seconds=60.0,
        backoff_multiplier=2.0,
        respect_retry_after=True,
        exc=RuntimeError("no header"),
    )
    assert 1.0 <= d <= 60.0
