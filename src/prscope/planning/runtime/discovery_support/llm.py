from __future__ import annotations

import asyncio
import json
from typing import Any

from ....config import LlmRetryConfig
from ....model_catalog import litellm_model_name, model_provider
from ....pricing import context_window_for_model
from ..llm_retry import (
    async_sleep,
    compute_retry_delay_seconds,
    is_non_chat_model_hint,
    is_transient_llm_failure,
)
from ..telemetry import completion_telemetry
from ..tools import CODEBASE_TOOLS


def _discovery_message_chars(messages: list[dict[str, Any]]) -> int:
    return sum(len(str(m.get("content", ""))) for m in messages)


def trim_discovery_messages_for_budget(
    messages: list[dict[str, Any]],
    max_chars: int,
) -> list[dict[str, Any]]:
    """Drop oldest tool messages until the transcript fits the budget (system + user preserved)."""
    if max_chars <= 0 or _discovery_message_chars(messages) <= max_chars:
        return messages
    out = list(messages)
    guard = 0
    while _discovery_message_chars(out) > max_chars and len(out) > 2 and guard < 1000:
        guard += 1
        removed = False
        for i in range(1, len(out)):
            if out[i].get("role") == "tool":
                del out[i]
                removed = True
                break
        if not removed:
            break
    return out


def _cap_tool_json_content(raw: str, max_chars: int) -> str:
    if len(raw) <= max_chars:
        return raw
    return json.dumps(
        {
            "truncated": True,
            "preview": raw[: max(0, max_chars - 240)],
            "note": "Tool JSON truncated for discovery context budget.",
        },
        ensure_ascii=False,
    )


class DiscoveryLLMClient:
    def __init__(self, manager: Any):
        self._manager = manager

    async def llm_call_with_tools(
        self,
        messages: list[dict[str, Any]],
        max_tool_rounds: int = 6,
        model_override: str | None = None,
    ) -> str:
        import litellm

        litellm.drop_params = True
        if hasattr(litellm, "set_verbose"):
            litellm.set_verbose = False
        session_id = str(getattr(self._manager, "_active_discovery_session_id", "default") or "default")
        conversation = list(messages)
        active_feature = self._manager._extract_feature_intent(self._manager._latest_user_message(messages))
        announced_scanning = False

        cfg = self._manager.config
        max_conv = max(8_192, int(getattr(cfg, "discovery_conversation_max_chars", 120_000)))
        per_tool_cap = max(4096, max_conv // 8)

        for _ in range(max_tool_rounds):
            before_len = len(conversation)
            before_chars = _discovery_message_chars(conversation)
            conversation = trim_discovery_messages_for_budget(conversation, max_conv)
            if len(conversation) < before_len or _discovery_message_chars(conversation) < before_chars:
                await self._manager._emit(
                    {
                        "type": "context_compaction",
                        "enabled": True,
                        "reason": "discovery_transcript_trim",
                        "session_stage": "discovery",
                    }
                )
            response = await self.safe_completion_call(
                litellm=litellm,
                messages=self._manager._normalize_roles(conversation),
                tools=CODEBASE_TOOLS,
                max_tokens=1800,
                model_override=model_override,
            )
            message = response.choices[0].message
            content = str(getattr(message, "content", None) or "").strip()
            tool_calls = getattr(message, "tool_calls", None) or []

            if tool_calls:
                if not announced_scanning:
                    announced_scanning = True
                    await self._manager._emit(
                        {
                            "type": "thinking",
                            "message": "Scanning codebase and refining questions...",
                        }
                    )
                conversation.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [
                            {
                                "id": getattr(tc, "id", None),
                                "type": "function",
                                "function": {
                                    "name": getattr(tc.function, "name", ""),
                                    "arguments": getattr(tc.function, "arguments", "{}"),
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )
                for tc in tool_calls:
                    tool_name = getattr(getattr(tc, "function", None), "name", "")
                    raw_args = getattr(getattr(tc, "function", None), "arguments", "{}") or "{}"
                    try:
                        parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else {}
                    except json.JSONDecodeError:
                        parsed_args = {}
                    path_hint = parsed_args.get("path") or parsed_args.get("relative_path")
                    query_hint = parsed_args.get("query") or parsed_args.get("pattern")
                    await self._manager._emit(
                        {
                            "type": "tool_call",
                            "name": tool_name,
                            "session_stage": "discovery",
                            "path": str(path_hint) if isinstance(path_hint, str) else None,
                            "query": str(query_hint) if isinstance(query_hint, str) else None,
                        }
                    )
                    try:
                        tool_started = asyncio.get_running_loop().time()
                        result = await asyncio.to_thread(self._manager.tool_executor.execute, tc)
                        tool_elapsed_ms = (asyncio.get_running_loop().time() - tool_started) * 1000.0
                    except Exception as exc:  # noqa: BLE001
                        tool_elapsed_ms = (asyncio.get_running_loop().time() - tool_started) * 1000.0
                        result = {
                            "tool_call_id": getattr(tc, "id", ""),
                            "name": "",
                            "result": {"error": str(exc)},
                        }
                    await self._manager._emit(
                        {
                            "type": "tool_result",
                            "name": tool_name,
                            "session_stage": "discovery",
                            "duration_ms": round(tool_elapsed_ms, 2),
                        }
                    )
                    await self._manager._ingest_feature_evidence_from_tool(
                        session_id=session_id,
                        feature=active_feature,
                        tool_name=tool_name,
                        parsed_args=parsed_args,
                        tool_result_payload=result["result"] if isinstance(result, dict) else {},
                    )
                    raw_content = json.dumps(result["result"])
                    raw_content = _cap_tool_json_content(raw_content, per_tool_cap)
                    conversation.append(
                        {
                            "role": "tool",
                            "tool_call_id": result["tool_call_id"],
                            "name": result["name"],
                            "content": raw_content,
                        }
                    )
                continue

            return content

        return "I've scanned the codebase. What aspect of this would you like to focus on first?"

    async def llm_call(self, messages: list[dict[str, Any]], model_override: str | None = None) -> str:
        import litellm

        litellm.drop_params = True
        if hasattr(litellm, "set_verbose"):
            litellm.set_verbose = False
        response = await self.safe_completion_call(
            litellm=litellm,
            messages=self._manager._normalize_roles(messages),
            max_tokens=900,
            model_override=model_override,
        )
        return str(response.choices[0].message.content or "").strip()

    async def safe_completion_call(
        self,
        *,
        litellm: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1200,
        model_override: str | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools is not None:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        cfg = self._manager.config
        policy = getattr(cfg, "llm_retry", None) or LlmRetryConfig()
        max_attempts = max(1, int(policy.max_attempts)) if policy.enabled else 1
        primary_model = model_override or cfg.author_model
        fallback_model = (
            str(getattr(cfg, "discovery_completion_fallback_model", None) or "gpt-4o-mini").strip() or "gpt-4o-mini"
        )
        models_to_try = [primary_model]
        if fallback_model != primary_model:
            models_to_try.append(fallback_model)

        last_error: Exception | None = None
        for idx, model in enumerate(models_to_try):
            litellm_model = litellm_model_name(model)
            fatal_for_models = False
            for attempt in range(max_attempts):
                try:
                    llm_started = asyncio.get_running_loop().time()
                    response = await asyncio.to_thread(
                        litellm.completion,
                        model=litellm_model,
                        **kwargs,
                    )
                    llm_elapsed_ms = (asyncio.get_running_loop().time() - llm_started) * 1000.0
                    telemetry = completion_telemetry(response, model=model)
                    context_window = context_window_for_model(model)
                    await self._manager._emit(
                        {
                            "type": "token_usage",
                            "session_stage": "discovery",
                            "model": model,
                            "model_provider": model_provider(model),
                            "prompt_tokens": telemetry.usage.prompt_tokens,
                            "completion_tokens": telemetry.usage.completion_tokens,
                            "call_cost_usd": telemetry.cost.total_cost_usd,
                            "llm_call_latency_ms": round(llm_elapsed_ms, 2),
                            "context_window_tokens": context_window,
                            "context_usage_ratio": (
                                round(float(telemetry.usage.prompt_tokens) / float(context_window), 4)
                                if context_window > 0
                                else None
                            ),
                        }
                    )
                    return response
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if is_non_chat_model_hint(exc) and idx < len(models_to_try) - 1:
                        break
                    if is_transient_llm_failure(exc) and attempt < max_attempts - 1 and policy.enabled:
                        delay = compute_retry_delay_seconds(
                            attempt_index=attempt,
                            initial_delay_seconds=policy.initial_delay_seconds,
                            max_delay_seconds=policy.max_delay_seconds,
                            backoff_multiplier=policy.backoff_multiplier,
                            respect_retry_after=policy.respect_retry_after,
                            exc=exc,
                        )
                        await self._manager._emit(
                            {
                                "type": "llm_retry",
                                "session_stage": "discovery",
                                "model": model,
                                "attempt": attempt + 1,
                                "max_attempts": max_attempts,
                                "delay_seconds": round(delay, 2),
                                "reason": "transient",
                            }
                        )
                        await async_sleep(delay)
                        continue
                    if not is_non_chat_model_hint(exc) or idx == len(models_to_try) - 1:
                        fatal_for_models = True
                        break

            if fatal_for_models:
                break

        if last_error is not None:
            raise RuntimeError(
                "Configured planning model is incompatible with chat completions. "
                "Update planning.author_model or use a chat-capable model."
            ) from last_error
        raise RuntimeError("Unknown completion failure during discovery.")
