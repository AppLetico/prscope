from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ....pricing import MODEL_CONTEXT_WINDOWS
from ..telemetry import completion_telemetry
from ..tools import CODEBASE_TOOLS


@dataclass
class FunctionCall:
    name: str
    arguments: str


@dataclass
class ToolCall:
    id: str
    function: FunctionCall


@dataclass
class Message:
    content: str
    tool_calls: list[ToolCall]


@dataclass
class Choice:
    message: Message


@dataclass
class ChatLikeResponse:
    choices: list[Choice]
    usage: Any | None = None


class AuthorLLMClient:
    def __init__(self, config: Any, event_emitter: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        self._config = config
        self._emit = event_emitter

    async def call(
        self,
        messages: list[dict[str, Any]],
        *,
        allow_tools: bool = True,
        max_output_tokens: int | None = None,
        model_override: str | None = None,
        timeout_seconds_override: int | Callable[[], int] | None = None,
    ) -> tuple[Any, str]:
        import litellm

        litellm.drop_params = True
        if hasattr(litellm, "set_verbose"):
            litellm.set_verbose = False
        return await self.safe_completion_call(
            litellm=litellm,
            messages=messages,
            allow_tools=allow_tools,
            max_output_tokens=max_output_tokens,
            model_override=model_override,
            timeout_seconds_override=timeout_seconds_override,
        )

    @staticmethod
    def prefer_responses_api(model: str) -> bool:
        return model.startswith("gpt-5")

    @staticmethod
    def responses_tools() -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for entry in CODEBASE_TOOLS:
            fn = entry.get("function", {})
            tools.append(
                {
                    "type": "function",
                    "name": str(fn.get("name", "")),
                    "description": str(fn.get("description", "")),
                    "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        return tools

    @staticmethod
    def as_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            if role == "tool":
                payload.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(message.get("tool_call_id", "")),
                        "output": str(message.get("content", "")),
                    }
                )
                continue
            if role == "assistant":
                content = str(message.get("content", ""))
                if content:
                    payload.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": content}],
                        }
                    )
                tool_calls = message.get("tool_calls", []) or []
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                        payload.append(
                            {
                                "type": "function_call",
                                "call_id": str(tc.get("id", "")) if isinstance(tc, dict) else "",
                                "name": str(fn.get("name", "")),
                                "arguments": str(fn.get("arguments", "{}")),
                            }
                        )
                continue
            payload.append(
                {
                    "role": role,
                    "content": [{"type": "input_text", "text": str(message.get("content", ""))}],
                }
            )
        return payload

    @staticmethod
    def extract_responses_text(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        output = getattr(response, "output", None)
        if isinstance(output, list):
            chunks: list[str] = []
            for item in output:
                content = getattr(item, "content", None)
                if not isinstance(content, list):
                    continue
                for part in content:
                    text = getattr(part, "text", None)
                    if isinstance(text, str) and text:
                        chunks.append(text)
            if chunks:
                return "\n".join(chunks).strip()
        return ""

    @staticmethod
    def extract_responses_tool_calls(response: Any) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        output = getattr(response, "output", None)
        if not isinstance(output, list):
            return tool_calls
        for item in output:
            item_type = str(getattr(item, "type", "") or "")
            if item_type != "function_call":
                continue
            name = str(getattr(item, "name", "") or "")
            arguments = str(getattr(item, "arguments", "") or "{}")
            call_id = str(getattr(item, "call_id", None) or getattr(item, "id", None) or f"call_{len(tool_calls)}")
            if not name:
                continue
            tool_calls.append(ToolCall(id=call_id, function=FunctionCall(name=name, arguments=arguments)))
        return tool_calls

    async def safe_completion_call(
        self,
        *,
        litellm: Any,
        messages: list[dict[str, Any]],
        allow_tools: bool,
        max_output_tokens: int | None,
        model_override: str | None,
        timeout_seconds_override: int | Callable[[], int] | None,
    ) -> tuple[Any, str]:
        fallback_model = "gpt-4o"
        configured_timeout_seconds = max(5, int(self._config.author_call_timeout_seconds))
        if callable(timeout_seconds_override):
            per_call_timeout_seconds = max(5, min(configured_timeout_seconds, int(timeout_seconds_override())))
        elif timeout_seconds_override is None:
            per_call_timeout_seconds = configured_timeout_seconds
        else:
            per_call_timeout_seconds = max(5, min(configured_timeout_seconds, int(timeout_seconds_override)))
        output_cap = max(512, min(4000, int(max_output_tokens or 4000)))
        primary_model = model_override or self._config.author_model
        models_to_try = [primary_model]
        if fallback_model != primary_model:
            models_to_try.append(fallback_model)

        last_error: Exception | None = None
        for idx, model in enumerate(models_to_try):
            try:
                if self.prefer_responses_api(model):
                    from openai import OpenAI

                    client = OpenAI()
                    responses_kwargs: dict[str, Any] = {
                        "model": model,
                        "input": self.as_responses_input(messages),
                        "max_output_tokens": output_cap,
                    }
                    if allow_tools:
                        responses_kwargs["tools"] = self.responses_tools()
                        responses_kwargs["tool_choice"] = "auto"
                    llm_started = asyncio.get_running_loop().time()
                    response = await asyncio.wait_for(
                        asyncio.to_thread(client.responses.create, **responses_kwargs),
                        timeout=per_call_timeout_seconds,
                    )
                    llm_elapsed_ms = (asyncio.get_running_loop().time() - llm_started) * 1000.0
                    text = self.extract_responses_text(response)
                    tool_calls = self.extract_responses_tool_calls(response)
                    chat_like = ChatLikeResponse(
                        choices=[Choice(message=Message(content=text, tool_calls=tool_calls))],
                        usage=getattr(response, "usage", None),
                    )
                    telemetry = completion_telemetry(response, model=model)
                    context_window = MODEL_CONTEXT_WINDOWS.get(model)
                    if model not in MODEL_CONTEXT_WINDOWS:
                        await self._emit(
                            {
                                "type": "warning",
                                "message": f"Unknown model '{model}' - context window tracking disabled",
                            }
                        )
                    if model not in MODEL_CONTEXT_WINDOWS:
                        await self._emit(
                            {
                                "type": "warning",
                                "message": f"Unknown model '{model}' - cost tracking disabled for this call",
                            }
                        )
                    await self._emit(
                        {
                            "type": "token_usage",
                            "session_stage": "author",
                            "model": model,
                            "prompt_tokens": telemetry.usage.prompt_tokens,
                            "completion_tokens": telemetry.usage.completion_tokens,
                            "call_cost_usd": telemetry.cost.total_cost_usd,
                            "llm_call_latency_ms": round(llm_elapsed_ms, 2),
                            "context_window_tokens": context_window,
                            "context_usage_ratio": (
                                round(float(telemetry.usage.prompt_tokens) / float(context_window), 4)
                                if context_window and context_window > 0
                                else None
                            ),
                        }
                    )
                    if context_window and telemetry.usage.prompt_tokens > int(context_window * 0.75):
                        await self._emit(
                            {
                                "type": "warning",
                                "message": (
                                    f"Prompt tokens {telemetry.usage.prompt_tokens} exceed "
                                    f"75% of context window ({context_window}) for {model}"
                                ),
                            }
                        )
                    return chat_like, model

                completion_kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": output_cap,
                }
                if allow_tools:
                    completion_kwargs["tools"] = CODEBASE_TOOLS
                    completion_kwargs["tool_choice"] = "auto"
                llm_started = asyncio.get_running_loop().time()
                response = await asyncio.wait_for(
                    asyncio.to_thread(litellm.completion, **completion_kwargs),
                    timeout=per_call_timeout_seconds,
                )
                llm_elapsed_ms = (asyncio.get_running_loop().time() - llm_started) * 1000.0
                telemetry = completion_telemetry(response, model=model)
                context_window = MODEL_CONTEXT_WINDOWS.get(model)
                if model not in MODEL_CONTEXT_WINDOWS:
                    await self._emit(
                        {
                            "type": "warning",
                            "message": f"Unknown model '{model}' - context window tracking disabled",
                        }
                    )
                if model not in MODEL_CONTEXT_WINDOWS:
                    await self._emit(
                        {
                            "type": "warning",
                            "message": f"Unknown model '{model}' - cost tracking disabled for this call",
                        }
                    )
                await self._emit(
                    {
                        "type": "token_usage",
                        "session_stage": "author",
                        "model": model,
                        "prompt_tokens": telemetry.usage.prompt_tokens,
                        "completion_tokens": telemetry.usage.completion_tokens,
                        "call_cost_usd": telemetry.cost.total_cost_usd,
                        "llm_call_latency_ms": round(llm_elapsed_ms, 2),
                        "context_window_tokens": context_window,
                        "context_usage_ratio": (
                            round(float(telemetry.usage.prompt_tokens) / float(context_window), 4)
                            if context_window and context_window > 0
                            else None
                        ),
                    }
                )
                if context_window and telemetry.usage.prompt_tokens > int(context_window * 0.75):
                    await self._emit(
                        {
                            "type": "warning",
                            "message": (
                                f"Prompt tokens {telemetry.usage.prompt_tokens} exceed "
                                f"75% of context window ({context_window}) for {model}"
                            ),
                        }
                    )
                return response, model
            except asyncio.TimeoutError as exc:
                last_error = RuntimeError(f"Model '{model}' timed out after {per_call_timeout_seconds}s")
                await self._emit(
                    {
                        "type": "warning",
                        "message": (
                            f"Author call timeout on {model} after {per_call_timeout_seconds}s; trying fallback model."
                        ),
                    }
                )
                if idx == len(models_to_try) - 1:
                    raise RuntimeError(
                        "Configured planning author model timed out and fallback also timed out."
                    ) from exc
                continue
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                err_text = str(exc).lower()
                non_chat_model = (
                    "not a chat model" in err_text
                    or "v1/chat/completions" in err_text
                    or "did you mean to use v1/completions" in err_text
                )
                if not non_chat_model or idx == len(models_to_try) - 1:
                    break

        if last_error is not None:
            raise RuntimeError(f"Configured planning author model failed. Last error: {last_error}") from last_error
        raise RuntimeError("Unknown completion failure during authoring.")
