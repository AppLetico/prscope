from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from prscope.model_catalog import litellm_model_name
from prscope.planning.runtime.transport.llm_client import AuthorLLMClient


def test_litellm_model_name_routes_google_models_to_gemini_provider() -> None:
    assert litellm_model_name("gemini-2.5-flash") == "gemini/gemini-2.5-flash"
    assert litellm_model_name("gpt-4o-mini") == "gpt-4o-mini"
    assert litellm_model_name("claude-haiku-4-5") == "claude-haiku-4-5"


@pytest.mark.asyncio
async def test_author_llm_client_routes_gemini_models_through_google_provider(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="draft plan", tool_calls=[]))],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=4),
    )

    def _completion(**kwargs):
        calls.append(kwargs)
        return fake_response

    fake_litellm = SimpleNamespace(completion=_completion, drop_params=False)
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    events: list[dict[str, object]] = []

    async def _emit(event: dict[str, object]) -> None:
        events.append(event)

    client = AuthorLLMClient(
        config=SimpleNamespace(author_call_timeout_seconds=15, author_model="gemini-2.5-flash"),
        event_emitter=_emit,
    )

    response, model = await client.call(
        [{"role": "user", "content": "Draft a localized implementation plan."}],
        allow_tools=False,
    )

    assert response.choices[0].message.content == "draft plan"
    assert model == "gemini-2.5-flash"
    assert calls[0]["model"] == "gemini/gemini-2.5-flash"
    assert events[0]["model"] == "gemini-2.5-flash"
