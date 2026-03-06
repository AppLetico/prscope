"""
Adversarial planning round pipeline orchestration.
"""

from __future__ import annotations

import time
from typing import Any

from ..author import AuthorResult
from .round_context import PlanningRoundContext


class AdversarialPlanningLoop:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    async def run_round(
        self, *, ctx: PlanningRoundContext, current_plan: Any, user_input: str | None
    ) -> tuple[Any, Any, Any]:
        if user_input:
            ctx.core.add_turn("user", user_input, round_number=ctx.round_number)

        async def emit_tool(
            name: str,
            status: str,
            stage: str = "planner",
            duration_ms: int | None = None,
            query: str | None = None,
        ) -> None:
            payload: dict[str, Any] = {
                "type": "tool_update",
                "tool": {"name": name, "status": status, "session_stage": stage},
            }
            if duration_ms is not None:
                payload["tool"]["duration_ms"] = duration_ms
            if query:
                payload["tool"]["query"] = query
            await self.runtime._emit_event(ctx.event_callback, payload, ctx.session_id)  # noqa: SLF001

        self.runtime.author.event_callback = lambda event: self.runtime._emit_event(  # noqa: SLF001
            ctx.event_callback, event, ctx.session_id
        )
        self.runtime.critic.event_callback = lambda event: self.runtime._emit_event(  # noqa: SLF001
            ctx.event_callback, event, ctx.session_id
        )

        review_result = await self.runtime._stage_design_review(  # noqa: SLF001
            ctx=ctx,
            current_plan_content=current_plan.plan_content,
            emit_tool=emit_tool,
        )
        current_plan_doc = self.runtime._plan_document_from_version(  # noqa: SLF001
            current_plan.plan_content, getattr(current_plan, "plan_json", None)
        )
        repair_plan = await self.runtime._stage_repair_plan(  # noqa: SLF001
            ctx=ctx,
            current_plan_doc=current_plan_doc,
            review_result=review_result,
            emit_tool=emit_tool,
        )
        updated_markdown, _, _ = await self.runtime._stage_revise_plan(  # noqa: SLF001
            ctx=ctx,
            current_plan_doc=current_plan_doc,
            repair_plan=repair_plan,
            review_result=review_result,
            emit_tool=emit_tool,
        )
        validation_review = await self.runtime._stage_validation_review(  # noqa: SLF001
            ctx=ctx,
            updated_markdown=updated_markdown,
            emit_tool=emit_tool,
        )
        implementability, convergence = await self.runtime._stage_convergence_check(  # noqa: SLF001
            ctx=ctx,
            updated_markdown=updated_markdown,
            validation_review=validation_review,
            emit_tool=emit_tool,
        )
        open_issue_count = int(convergence.major_issues)

        self.runtime.store.add_round_metrics(
            session_id=ctx.session_id,
            repo_name=self.runtime.repo.name,
            round_number=ctx.round_number,
            author_prompt_tokens=ctx.state.author_prompt_tokens,
            author_completion_tokens=ctx.state.author_completion_tokens,
            critic_prompt_tokens=ctx.state.critic_prompt_tokens,
            critic_completion_tokens=ctx.state.critic_completion_tokens,
            max_prompt_tokens=ctx.state.max_prompt_tokens,
            major_issues=open_issue_count,
            minor_issues=0,
            critic_confidence=validation_review.design_quality_score / 10.0,
            vagueness_score=0.0,
            citation_count=0,
            constraint_violations=validation_review.constraint_violations,
            resolved_since_last_round=validation_review.resolved_issues,
            clarifications_this_round=0,
            call_cost_usd=ctx.state.round_cost_usd,
            issues_resolved=len(validation_review.resolved_issues),
            issues_introduced=len(validation_review.blocking_issues),
            net_improvement=len(validation_review.resolved_issues) - len(validation_review.blocking_issues),
            model_costs={},
            time_to_first_tool_call=None,
            grounding_ratio=None,
            static_injection_tokens_pct=None,
            rejected_for_no_discovery=0,
            rejected_for_grounding=0,
            rejected_for_budget=0,
            average_read_depth_per_round=None,
            time_between_tool_calls=None,
            rejection_reasons=[],
            plan_quality_score=validation_review.design_quality_score / 10.0,
            unsupported_claims_count=0,
            missing_evidence_count=0,
        )

        final_status = "converged" if convergence.converged else "refining"
        snapshot = ctx.core.transition_and_snapshot(final_status, phase_message=None, allow_round_stability=True)
        await self.runtime._emit_event(ctx.event_callback, snapshot, ctx.session_id)  # noqa: SLF001
        await self.runtime._emit_event(  # noqa: SLF001
            ctx.event_callback,
            {
                "type": "plan_ready",
                "round": ctx.round_number,
                "saved_at_unix_s": time.time(),
            },
            ctx.session_id,
        )
        if ctx.event_callback:
            await self.runtime._emit_event(  # noqa: SLF001
                ctx.event_callback,
                {"type": "complete", "message": "Adversarial round complete"},
                ctx.session_id,
            )

        author_result = AuthorResult(
            plan=updated_markdown,
            unverified_references=set(),
            accessed_paths=self.runtime._session_reads(ctx.session_id).copy(),  # noqa: SLF001
        )
        ctx.state.constraint_eval = validation_review
        ctx.state.issue_tracker = ctx.issue_tracker
        self.runtime._persist_state_snapshot(ctx.session_id)  # noqa: SLF001
        return validation_review, author_result, convergence
