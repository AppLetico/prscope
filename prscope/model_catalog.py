"""
Provider-aware model catalog and key-based availability checks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .pricing import MODEL_CONTEXT_WINDOWS


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model_id: str
    # Each tuple entry is a group of equivalent env keys (any-of).
    required_env_groups: tuple[tuple[str, ...], ...]


CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec("openai", "gpt-4o-mini", (("OPENAI_API_KEY",),)),
    ModelSpec("openai", "gpt-4o", (("OPENAI_API_KEY",),)),
    ModelSpec("openai", "gpt-5.2", (("OPENAI_API_KEY",),)),
    ModelSpec("openai", "gpt-5.3-codex", (("OPENAI_API_KEY",),)),
    ModelSpec("openai", "o3", (("OPENAI_API_KEY",),)),
    ModelSpec("openai", "o4-mini", (("OPENAI_API_KEY",),)),
    ModelSpec("anthropic", "claude-3-haiku-20240307", (("ANTHROPIC_API_KEY",),)),
    ModelSpec("anthropic", "claude-3-5-sonnet-20241022", (("ANTHROPIC_API_KEY",),)),
    ModelSpec("anthropic", "claude-sonnet-4-20250514", (("ANTHROPIC_API_KEY",),)),
    ModelSpec("anthropic", "claude-opus-4-20250514", (("ANTHROPIC_API_KEY",),)),
    ModelSpec("anthropic", "claude-sonnet-4-5", (("ANTHROPIC_API_KEY",),)),
    ModelSpec("anthropic", "claude-opus-4-5", (("ANTHROPIC_API_KEY",),)),
    ModelSpec("anthropic", "claude-sonnet-4-6", (("ANTHROPIC_API_KEY",),)),
    ModelSpec("anthropic", "claude-opus-4-6", (("ANTHROPIC_API_KEY",),)),
    ModelSpec("google", "gemini-2.0-flash", (("GOOGLE_API_KEY", "GEMINI_API_KEY"),)),
    ModelSpec("google", "gemini-1.5-pro", (("GOOGLE_API_KEY", "GEMINI_API_KEY"),)),
    ModelSpec("google", "gemini-2.5-flash-lite", (("GOOGLE_API_KEY", "GEMINI_API_KEY"),)),
    ModelSpec("google", "gemini-2.5-flash", (("GOOGLE_API_KEY", "GEMINI_API_KEY"),)),
    ModelSpec("google", "gemini-2.5-pro", (("GOOGLE_API_KEY", "GEMINI_API_KEY"),)),
    ModelSpec("google", "gemini-3-flash", (("GOOGLE_API_KEY", "GEMINI_API_KEY"),)),
    ModelSpec("google", "gemini-3-pro", (("GOOGLE_API_KEY", "GEMINI_API_KEY"),)),
    ModelSpec("google", "gemini-3.1-pro", (("GOOGLE_API_KEY", "GEMINI_API_KEY"),)),
)


def _group_missing_reason(group: tuple[str, ...]) -> str:
    if len(group) == 1:
        return f"missing {group[0]}"
    return "missing one of: " + ", ".join(group)


def _group_available(group: tuple[str, ...]) -> bool:
    return any(bool(os.environ.get(key)) for key in group)


def list_models() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for spec in CATALOG:
        missing_groups = [group for group in spec.required_env_groups if not _group_available(group)]
        available = len(missing_groups) == 0
        reason = _group_missing_reason(missing_groups[0]) if missing_groups else None
        required_keys: list[str] = []
        for group in spec.required_env_groups:
            required_keys.extend(group)
        items.append(
            {
                "provider": spec.provider,
                "model_id": spec.model_id,
                "available": available,
                "unavailable_reason": reason,
                "required_env_keys": sorted(set(required_keys)),
                "context_window_tokens": MODEL_CONTEXT_WINDOWS.get(spec.model_id),
            }
        )
    return items


def get_model(model_id: str) -> dict[str, Any] | None:
    for item in list_models():
        if item["model_id"] == model_id:
            return item
    return None
