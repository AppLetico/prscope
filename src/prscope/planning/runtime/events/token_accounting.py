"""
Token usage accounting helpers for planning runtime events.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ....store import Store
    from ..state import PlanningState


def apply_token_usage_event(
    *,
    state: PlanningState,
    store: Store,
    session_id: str,
    event: dict[str, Any],
) -> dict[str, Any]:
    """
    Apply token/cost accounting for a token_usage event and return enriched payload.
    """
    call_cost = float(event.get("call_cost_usd", 0.0) or 0.0)
    prompt_tokens = int(event.get("prompt_tokens", 0) or 0)
    completion_tokens = int(event.get("completion_tokens", 0) or 0)
    stage = str(event.get("session_stage", "")).strip().lower()
    running_cost = float(state.session_cost_usd) + call_cost
    max_prompt = max(int(state.max_prompt_tokens), prompt_tokens)

    state.round_cost_usd = float(state.round_cost_usd) + call_cost
    if stage == "author":
        state.author_prompt_tokens += prompt_tokens
        state.author_completion_tokens += completion_tokens
    else:
        state.critic_prompt_tokens += prompt_tokens
        state.critic_completion_tokens += completion_tokens
    state.session_cost_usd = running_cost
    state.max_prompt_tokens = max_prompt

    store.update_planning_session(
        session_id,
        session_total_cost_usd=running_cost,
        max_prompt_tokens=max_prompt,
    )

    return {
        **event,
        "session_total_usd": running_cost,
        "max_prompt_tokens": max_prompt,
        "round_cost_usd": state.round_cost_usd,
    }
