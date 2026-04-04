"""
Adversarial planning round pipeline orchestration.
"""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from ..author import AuthorResult
from ..critic import ReviewResult, skipped_validation_review_placeholder, synthetic_review_for_chat_refinement
from ..reasoning.refinement_reasoner import RefinementReasoner
from .round_context import PlanningRoundContext


class AdversarialPlanningLoop:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def _emit_tool_factory(self, ctx: PlanningRoundContext) -> Callable[..., Awaitable[None]]:
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

        return emit_tool

    @staticmethod
    def _resolve_plan_panel_tracked_issue_ids_when_validation_skipped(ctx: PlanningRoundContext) -> None:
        """Close issues named in PlanPanel prompts when validation_review was skipped (no critic JSON).

        Without this, batch \"Add all to chat\" + skip_validation_review leaves issue_* open even after
        a successful repair/revise because only validation_review used to call resolve_issue.
        """
        req = str(ctx.requirements or "")
        if not RefinementReasoner.looks_like_plan_panel_issue_followup(req):
            return
        tracker = ctx.issue_tracker
        open_fn = getattr(tracker, "open_issues", None)
        resolve_fn = getattr(tracker, "resolve_issue", None)
        if not callable(open_fn) or not callable(resolve_fn):
            return
        asked_raw = re.findall(r"\b(issue_\d+)\b", req, flags=re.IGNORECASE)
        if not asked_raw:
            return
        canon_fn = getattr(tracker, "canonical_issue_id", None)
        open_ids = {str(getattr(i, "id", "")).strip() for i in open_fn() if str(getattr(i, "id", "")).strip()}
        for raw in asked_raw:
            key = canon_fn(raw) if callable(canon_fn) else raw
            if key not in open_ids and raw not in open_ids:
                continue
            rid = key if key in open_ids else raw
            resolve_fn(
                rid,
                ctx.round_number,
                propagate_causes=False,
                resolution_source="lightweight",
            )

    def _wire_model_callbacks(self, ctx: PlanningRoundContext) -> None:
        self.runtime.author.event_callback = lambda event: self.runtime._emit_event(  # noqa: SLF001
            ctx.event_callback, event, ctx.session_id
        )
        self.runtime.critic.event_callback = lambda event: self.runtime._emit_event(  # noqa: SLF001
            ctx.event_callback, event, ctx.session_id
        )

    async def _complete_author_phases(
        self,
        *,
        ctx: PlanningRoundContext,
        current_plan: Any,
        review_result: ReviewResult,
        emit_tool: Callable[..., Awaitable[None]],
        skip_validation_review: bool = False,
    ) -> tuple[ReviewResult, AuthorResult, Any]:
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
        if skip_validation_review:
            validation_review = skipped_validation_review_placeholder()
            await emit_tool(
                "review_validation",
                "done",
                stage="reviewer",
                duration_ms=0,
                query="Deferred — run Review for critic validation",
            )
        else:
            validation_review = await self.runtime._stage_validation_review(  # noqa: SLF001
                ctx=ctx,
                updated_markdown=updated_markdown,
                emit_tool=emit_tool,
            )
        if skip_validation_review:
            self._resolve_plan_panel_tracked_issue_ids_when_validation_skipped(ctx)
        _implementability, convergence = await self.runtime._stage_convergence_check(  # noqa: SLF001
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
        self.runtime.store.update_planning_session(ctx.session_id, critique_pending_apply=0)
        self.runtime._persist_state_snapshot(ctx.session_id)  # noqa: SLF001
        return validation_review, author_result, convergence

    async def run_round(
        self, *, ctx: PlanningRoundContext, current_plan: Any, user_input: str | None
    ) -> tuple[Any, Any, Any]:
        if user_input:
            # Avoid duplicate user rows when lightweight edit partially persisted the same message
            # before failing over to this full round.
            def _norm_msg(s: str) -> str:
                return " ".join(str(s).strip().split())

            turns = ctx.core.get_conversation()
            last = turns[-1] if turns else None
            already = last is not None and last.role == "user" and _norm_msg(last.content) == _norm_msg(user_input)
            if not already:
                ctx.core.add_turn("user", user_input, round_number=ctx.round_number)

        await self.runtime._prepare_adversarial_compaction_context(  # noqa: SLF001
            ctx=ctx,
            plan_content=str(getattr(current_plan, "plan_content", "") or ""),
        )

        emit_tool = self._emit_tool_factory(ctx)
        self._wire_model_callbacks(ctx)

        st = ctx.state
        # Chat-driven refinement must NOT run a new critic design_review (Review button only).
        # Reuse stored critique when present; otherwise derive blocking context from issues + chat.
        skip_design_review_for_chat = bool(user_input)
        if skip_design_review_for_chat:
            sess = ctx.core.get_session()
            if st.review is not None:
                phase_msg = "Applying feedback to the plan (reusing last design review)"
            else:
                phase_msg = "Applying chat feedback to the plan (no new design review)"
            if ctx.round_number <= sess.current_round:
                snapshot = ctx.core.transition_and_snapshot(
                    "refining",
                    phase_message=phase_msg,
                    allow_round_stability=True,
                )
            else:
                snapshot = ctx.core.transition_and_snapshot(
                    "refining",
                    phase_message=phase_msg,
                    current_round=ctx.round_number,
                )
            await self.runtime._emit_event(ctx.event_callback, snapshot, ctx.session_id)  # noqa: SLF001
            await emit_tool("design_review", "done", stage="reviewer", duration_ms=0)
            if st.review is not None:
                review_result = st.review
            else:
                review_result = synthetic_review_for_chat_refinement(
                    ctx.issue_tracker,
                    user_input=user_input or "",
                )
        else:
            review_result = await self.runtime._stage_design_review(  # noqa: SLF001
                ctx=ctx,
                current_plan_content=current_plan.plan_content,
                emit_tool=emit_tool,
            )
        return await self._complete_author_phases(
            ctx=ctx,
            current_plan=current_plan,
            review_result=review_result,
            emit_tool=emit_tool,
            skip_validation_review=ctx.skip_validation_review,
        )

    async def run_critique_only(self, *, ctx: PlanningRoundContext, current_plan: Any) -> ReviewResult:
        """Run design review only; leave session idle with critique_pending_apply set."""
        await self.runtime._prepare_adversarial_compaction_context(  # noqa: SLF001
            ctx=ctx,
            plan_content=str(getattr(current_plan, "plan_content", "") or ""),
        )

        emit_tool = self._emit_tool_factory(ctx)
        self._wire_model_callbacks(ctx)

        review_result = await self.runtime._stage_design_review(  # noqa: SLF001
            ctx=ctx,
            current_plan_content=current_plan.plan_content,
            emit_tool=emit_tool,
        )
        snapshot = ctx.core.transition_and_snapshot("refining", phase_message=None, allow_round_stability=True)
        await self.runtime._emit_event(ctx.event_callback, snapshot, ctx.session_id)  # noqa: SLF001
        self.runtime.store.update_planning_session(ctx.session_id, critique_pending_apply=1)
        ctx.state.issue_tracker = ctx.issue_tracker
        self.runtime._persist_state_snapshot(ctx.session_id)  # noqa: SLF001
        if ctx.event_callback:
            await self.runtime._emit_event(  # noqa: SLF001
                ctx.event_callback,
                {"type": "complete", "message": "Critique complete — apply revision when ready"},
                ctx.session_id,
            )
        return review_result

    async def continue_after_critique(self, *, ctx: PlanningRoundContext, current_plan: Any) -> tuple[Any, Any, Any]:
        """Apply repair/revise after a stored critique (no second design review).

        Critic validation_review is skipped so the user controls when the next critic
        pass runs (Review button). Convergence uses a placeholder ReviewResult.
        """
        if ctx.state.review is None:
            raise ValueError("No critique is available to apply; run a review first.")
        await self.runtime._prepare_adversarial_compaction_context(  # noqa: SLF001
            ctx=ctx,
            plan_content=str(getattr(current_plan, "plan_content", "") or ""),
        )

        emit_tool = self._emit_tool_factory(ctx)
        self._wire_model_callbacks(ctx)

        snapshot = ctx.core.transition_and_snapshot(
            "refining",
            phase_message="Applying critique to the plan",
            allow_round_stability=True,
        )
        await self.runtime._emit_event(ctx.event_callback, snapshot, ctx.session_id)  # noqa: SLF001
        await emit_tool("design_review", "done", stage="reviewer", duration_ms=0)

        ctx.state.defer_validation_after_apply = True
        try:
            return await self._complete_author_phases(
                ctx=ctx,
                current_plan=current_plan,
                review_result=ctx.state.review,
                emit_tool=emit_tool,
                skip_validation_review=True,
            )
        finally:
            ctx.state.defer_validation_after_apply = False
