"""
Prior-critique compaction for adversarial refinement (working_summary).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from ....config import PlanningConfig
from ....pricing import context_window_for_model
from ..author import AuthorAgent
from ..context import CritiqueCompressor
from ..pipeline.round_context import PlanningRoundContext
from ..state import PlanningState


class AdversarialCompaction:
    """Encapsulates critique transcript compaction for long refinement rounds."""

    def __init__(
        self,
        *,
        planning_config: PlanningConfig,
        compressor: CritiqueCompressor,
        state_getter: Callable[[str], PlanningState],
        author: AuthorAgent,
        emit_event: Callable[..., Awaitable[None]],
        single_line: Callable[..., str],
    ) -> None:
        self._planning_config = planning_config
        self._compressor = compressor
        self._state_getter = state_getter
        self._author = author
        self._emit_event = emit_event
        self._single_line = single_line

    @staticmethod
    def extract_critic_turns(turns: list[Any]) -> list[str]:
        return [t.content for t in turns if t.role == "critic" and t.content]

    @staticmethod
    def context_window_limit(config: PlanningConfig) -> int:
        windows = [context_window_for_model(m) for m in {config.author_model, config.critic_model}]
        return min(windows) if windows else context_window_for_model(config.author_model)

    def should_compact_context(
        self,
        *,
        session_id: str,
        round_number: int,
        critic_turns: list[str],
        current_plan: str,
    ) -> bool:
        if round_number < 2:
            return False
        if len(critic_turns) >= 4:
            return True
        context_window = self.context_window_limit(self._planning_config)
        recent_peak = int(self._state_getter(session_id).max_prompt_tokens or 0)
        if recent_peak > int(context_window * 0.65):
            return True
        if len(current_plan) >= 14_000:
            return True
        return False

    def build_working_summary(
        self,
        *,
        requirements: str,
        critic_turns: list[str],
        current_plan: str,
    ) -> str:
        critique_summary = "(none yet)"
        if critic_turns:
            try:
                critique_summary = self._compressor.summarize(critic_turns)
            except Exception:  # noqa: BLE001
                critique_summary = critic_turns[-1][:1200]
        objective = self._single_line(requirements or "Refine plan against critique.", limit=280)
        summary = (
            "WORKING SUMMARY (compact prior rounds)\n\n"
            f"Objective: {objective}\n\n"
            f"Current plan snapshot (excerpt):\n{current_plan[:1800]}\n\n"
            f"Prior critique trajectory:\n{critique_summary}"
        )
        return summary[:5000]

    async def summarize_critiques_for_compaction(
        self,
        *,
        session_id: str,
        critic_turns: list[str],
        requirements: str,
        current_plan: str,
        event_callback: Any | None = None,
    ) -> str:
        cfg = self._planning_config
        state = self._state_getter(session_id)
        critique_summary: str | None = None
        if (
            cfg.critique_llm_summarize_enabled
            and critic_turns
            and int(state.max_prompt_tokens or 0) >= cfg.critique_llm_summarize_prompt_tokens_threshold
        ):
            model = cfg.critique_llm_summarize_model or cfg.critic_model
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Summarize prior design critiques in under 800 tokens. "
                        "Preserve constraint IDs, blocking issues, and recommended changes. "
                        "Plain text only, no markdown fences."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Requirements (one line): {self._single_line(requirements, limit=400)}\n\n"
                        f"Critiques (newest last):\n\n" + "\n\n---\n\n".join(critic_turns[-6:])
                    ),
                },
            ]
            try:
                response, _ = await self._author._llm_client.call(
                    messages,
                    allow_tools=False,
                    max_output_tokens=800,
                    model_override=model,
                )
                text = str(response.choices[0].message.content or "").strip()
                if text:
                    critique_summary = text
            except Exception:  # noqa: BLE001
                critique_summary = None
        if critique_summary is None:
            out = self.build_working_summary(
                requirements=requirements,
                critic_turns=critic_turns,
                current_plan=current_plan,
            )
            if event_callback is not None:
                await self._emit_event(
                    event_callback,
                    {
                        "type": "context_compaction",
                        "enabled": True,
                        "reason": "critique_heuristic_summary",
                        "session_stage": "refinement",
                    },
                    session_id,
                )
            return out
        objective = self._single_line(requirements or "Refine plan against critique.", limit=280)
        summary = (
            "WORKING SUMMARY (compact prior rounds)\n\n"
            f"Objective: {objective}\n\n"
            f"Current plan snapshot (excerpt):\n{current_plan[:1800]}\n\n"
            f"Prior critique trajectory (LLM summary):\n{critique_summary}"
        )
        if event_callback is not None:
            await self._emit_event(
                event_callback,
                {
                    "type": "context_compaction",
                    "enabled": True,
                    "reason": "critique_llm_summary",
                    "session_stage": "refinement",
                },
                session_id,
            )
        return summary[:5000]

    async def prepare_adversarial_compaction_context(
        self,
        *,
        ctx: PlanningRoundContext,
        plan_content: str,
    ) -> None:
        critic_turns = self.extract_critic_turns(ctx.core.get_conversation())
        text = str(plan_content or "")
        if self.should_compact_context(
            session_id=ctx.session_id,
            round_number=ctx.round_number,
            critic_turns=critic_turns,
            current_plan=text,
        ):
            ctx.state.working_summary = await self.summarize_critiques_for_compaction(
                session_id=ctx.session_id,
                critic_turns=critic_turns,
                requirements=ctx.requirements,
                current_plan=text,
                event_callback=ctx.event_callback,
            )
        else:
            ctx.state.working_summary = ""
