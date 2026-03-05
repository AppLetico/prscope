"""
Stage implementations for adversarial planning rounds.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, Literal

from ...core import ConvergenceResult
from ..author import PlanDocument, RepairPlan, RevisionResult, apply_section_updates, render_markdown
from ..critic import ImplementabilityResult, ReviewResult
from ..review import ManifestoCheckResult
from .round_context import PlanningRoundContext


class PlanningStages:
    def __init__(
        self,
        *,
        emit_event: Callable[[Any | None, dict[str, Any], str], Any],
        repo_memory: Callable[[Any], dict[str, str]],
        critic: Any,
        author: Any,
        design_record_payload: Callable[[Any], dict[str, Any] | None],
        design_record_from_payload: Callable[[dict[str, Any] | None], Any],
        review_chat_summary: Callable[[ReviewResult], str],
        manifesto_checker: Any,
        causality_extractor: Any | None = None,
    ) -> None:
        self._emit_event = emit_event
        self._repo_memory = repo_memory
        self._critic = critic
        self._author = author
        self._design_record_payload = design_record_payload
        self._design_record_from_payload = design_record_from_payload
        self._review_chat_summary = review_chat_summary
        self._manifesto_checker = manifesto_checker
        self._causality_extractor = causality_extractor

    async def design_review(
        self,
        *,
        ctx: PlanningRoundContext,
        current_plan_content: str,
        emit_tool: Callable[..., Any],
    ) -> ReviewResult:
        snapshot = ctx.core.transition_and_snapshot(
            "refining",
            phase_message="Running design review",
            current_round=ctx.round_number,
        )
        await self._emit_event(ctx.event_callback, snapshot, ctx.session_id)
        review_mode: Literal["initial", "validation", "stabilization", "implementability"] = "initial"
        architecture_change_count = int(ctx.state.architecture_change_count)
        if ctx.round_number > 0 and architecture_change_count / float(ctx.round_number) > 0.7:
            review_mode = "stabilization"
        await emit_tool("design_review", "running", stage="reviewer")
        review_started = time.perf_counter()
        blocks = self._repo_memory(ctx.state)
        review_payload = await self._critic.run_design_review(
            requirements=ctx.requirements,
            plan_content=current_plan_content,
            manifesto=ctx.state.manifesto,
            architecture=blocks.get("architecture", ""),
            design_record=json.dumps(self._design_record_payload(ctx.state.design_record) or {}, indent=2),
            modules=blocks.get("modules", ""),
            patterns=blocks.get("patterns", ""),
            constraints=ctx.state.constraints,
            prior_critique=ctx.issue_tracker.distilled_context(),
            model_override=ctx.selected_critic_model,
            session_id=ctx.session_id,
            round_number=ctx.round_number,
            mode=review_mode,
        )
        if not isinstance(review_payload, ReviewResult):
            raise RuntimeError("Expected ReviewResult from design review phase")
        review_result = review_payload
        ctx.state.review = review_result
        await emit_tool(
            "design_review",
            "done",
            stage="reviewer",
            duration_ms=round((time.perf_counter() - review_started) * 1000),
        )
        ctx.core.add_turn(
            "critic",
            self._review_chat_summary(review_result),
            round_number=ctx.round_number,
            parse_error=review_result.parse_error,
        )
        for item in [*review_result.blocking_issues, *review_result.architectural_concerns]:
            if item.strip():
                ctx.issue_tracker.add_issue(item, ctx.round_number)
        primary_issue_id = ""
        if review_result.primary_issue and review_result.primary_issue.strip():
            primary_issue_id = ctx.issue_tracker.add_issue(
                review_result.primary_issue,
                ctx.round_number,
                source="critic",
            ).id
            for derived_item in [*review_result.blocking_issues, *review_result.architectural_concerns]:
                if not derived_item.strip():
                    continue
                derived = ctx.issue_tracker.add_issue(derived_item, ctx.round_number, source="critic")
                if primary_issue_id and derived.id:
                    ctx.issue_tracker.add_edge(primary_issue_id, derived.id, "causes")
        for constraint_id in review_result.constraint_violations:
            constraint_issue = ctx.issue_tracker.add_issue(
                f"Constraint violation: {constraint_id}",
                ctx.round_number,
                source="critic",
            )
            if primary_issue_id and constraint_issue.id:
                ctx.issue_tracker.add_edge(primary_issue_id, constraint_issue.id, "depends_on")
        for issue_id in review_result.resolved_issues:
            ctx.issue_tracker.resolve_issue(issue_id, ctx.round_number)
        if self._causality_extractor is not None:
            self._causality_extractor.extract_edges(
                graph=ctx.issue_tracker,
                review=review_result,
                round_number=ctx.round_number,
            )
        return review_result

    async def repair_plan(
        self,
        *,
        ctx: PlanningRoundContext,
        current_plan_doc: PlanDocument,
        review_result: ReviewResult,
        emit_tool: Callable[..., Any],
    ) -> RepairPlan:
        snapshot = ctx.core.transition_and_snapshot(
            "refining", phase_message="Planning repair", allow_round_stability=True
        )
        await self._emit_event(ctx.event_callback, snapshot, ctx.session_id)
        await emit_tool("repair_planning", "running", stage="author")
        repair_started = time.perf_counter()
        design_record_payload = self._design_record_payload(ctx.state.design_record)
        if design_record_payload:
            try:
                updated_record = await self._author.update_design_record(
                    design_record=design_record_payload,
                    review=review_result,
                    requirements=ctx.requirements,
                    model_override=ctx.selected_author_model,
                )
                ctx.state.design_record = self._design_record_from_payload(updated_record)
            except Exception as exc:  # noqa: BLE001
                await self._emit_event(
                    ctx.event_callback,
                    {
                        "type": "warning",
                        "message": f"Design record update failed; continuing with existing record: {exc}",
                    },
                    ctx.session_id,
                )
        repair_plan = await self._author.plan_repair(
            review=review_result,
            plan=current_plan_doc,
            requirements=ctx.requirements,
            design_record=self._design_record_payload(ctx.state.design_record),
            model_override=ctx.selected_author_model,
        )
        must_fix_count = 2
        must_fix_issues: list[str] = []
        if review_result.issue_priority:
            must_fix_issues = [item for item in review_result.issue_priority[:must_fix_count] if item.strip()]
        elif review_result.primary_issue:
            must_fix_issues = [review_result.primary_issue]
        accepted = {item.strip() for item in repair_plan.accepted_issues}
        rejected = {item.strip() for item in repair_plan.rejected_issues}
        missing_must_fix = [item for item in must_fix_issues if item not in accepted and item not in rejected]
        if missing_must_fix:
            repair_plan = await self._author.plan_repair(
                review=review_result,
                plan=current_plan_doc,
                requirements=(
                    f"{ctx.requirements}\n\n"
                    "You MUST address or explicitly reject each must-fix issue with rationale:\n"
                    + "\n".join(f"- {item}" for item in missing_must_fix)
                ),
                design_record=self._design_record_payload(ctx.state.design_record),
                model_override=ctx.selected_author_model,
            )
            accepted = {item.strip() for item in repair_plan.accepted_issues}
            rejected = {item.strip() for item in repair_plan.rejected_issues}
            unresolved_must_fix = [item for item in must_fix_issues if item not in accepted and item not in rejected]
            for unresolved in unresolved_must_fix:
                ctx.issue_tracker.add_issue(f"Must-fix unresolved without rationale: {unresolved}", ctx.round_number)
                await self._emit_event(
                    ctx.event_callback,
                    {
                        "type": "warning",
                        "message": (
                            f"Must-fix issue remained unresolved without explicit rejection rationale: {unresolved}"
                        ),
                    },
                    ctx.session_id,
                )
        await emit_tool(
            "repair_planning",
            "done",
            stage="author",
            duration_ms=round((time.perf_counter() - repair_started) * 1000),
        )
        ctx.core.add_turn(
            "author",
            (
                "Repair planning complete.\n\n"
                f"Problem understanding: {repair_plan.problem_understanding}\n"
                "Target sections: "
                f"{', '.join(repair_plan.target_sections) if repair_plan.target_sections else '(none)'}\n"
                f"Repair strategy: {repair_plan.repair_strategy}"
            ),
            round_number=ctx.round_number,
        )
        return repair_plan

    async def revise_plan(
        self,
        *,
        ctx: PlanningRoundContext,
        current_plan_doc: PlanDocument,
        repair_plan: RepairPlan,
        review_result: ReviewResult,
        emit_tool: Callable[..., Any],
    ) -> tuple[str, list[str], RevisionResult]:
        snapshot = ctx.core.transition_and_snapshot(
            "refining", phase_message="Revising design", allow_round_stability=True
        )
        await self._emit_event(ctx.event_callback, snapshot, ctx.session_id)
        await emit_tool("apply_critique", "running", stage="author")
        open_issues = ctx.issue_tracker.open_issues()
        revision_budget = 1 if review_result.primary_issue else min(3, max(1, len(open_issues)))
        revise_started = time.perf_counter()
        revision_result = await self._author.revise_plan(
            repair_plan=repair_plan,
            current_plan=current_plan_doc,
            requirements=ctx.requirements,
            design_record=self._design_record_payload(ctx.state.design_record),
            revision_budget=revision_budget,
            model_override=ctx.selected_author_model,
            simplest_possible_design=review_result.simplest_possible_design,
        )
        updated_plan = apply_section_updates(current_plan_doc, revision_result.updates)
        updated_markdown = render_markdown(updated_plan)
        changed_sections = sorted(list(revision_result.updates.keys()))
        ctx.core.save_plan_version(
            updated_markdown,
            round_number=ctx.round_number,
            plan_document=updated_plan,
            changed_sections=changed_sections,
        )
        ctx.state.plan_markdown = updated_markdown
        architecture_changed = any(section in {"architecture", "files_changed"} for section in changed_sections)
        ctx.state.architecture_change_count = int(ctx.state.architecture_change_count) + int(architecture_changed)
        ctx.state.architecture_change_rounds.append(architecture_changed)
        if len(ctx.state.architecture_change_rounds) > 8:
            ctx.state.architecture_change_rounds = ctx.state.architecture_change_rounds[-8:]
        summary_bits: list[str] = []
        if review_result.primary_issue:
            summary_bits.append(f"Primary issue: {review_result.primary_issue}")
        if changed_sections:
            summary_bits.append(f"Updated: {', '.join(changed_sections)}")
        await emit_tool(
            "apply_critique",
            "done",
            stage="author",
            duration_ms=round((time.perf_counter() - revise_started) * 1000),
            query=" | ".join(summary_bits) if summary_bits else None,
        )
        ctx.core.add_turn(
            "author",
            (
                f"Updated sections: {', '.join(changed_sections) if changed_sections else '(none)'}\n\n"
                f"Problem understanding: {revision_result.problem_understanding}\n"
                f"Review prediction: {revision_result.review_prediction}"
            ),
            round_number=ctx.round_number,
        )
        return updated_markdown, changed_sections, revision_result

    async def validation_review(
        self,
        *,
        ctx: PlanningRoundContext,
        updated_markdown: str,
        emit_tool: Callable[..., Any],
    ) -> ReviewResult:
        snapshot = ctx.core.transition_and_snapshot(
            "refining", phase_message="Validating changes", allow_round_stability=True
        )
        await self._emit_event(ctx.event_callback, snapshot, ctx.session_id)
        manifesto_result: ManifestoCheckResult = self._manifesto_checker.validate(
            plan_content=updated_markdown,
            constraints=ctx.state.constraints,
        )
        if manifesto_result.violations or manifesto_result.warnings:
            await self._emit_event(
                ctx.event_callback,
                {
                    "type": "warning",
                    "message": (
                        "Manifesto check warnings: "
                        + "; ".join([*manifesto_result.violations, *manifesto_result.warnings][:4])
                    ),
                },
                ctx.session_id,
            )
        await emit_tool("review_validation", "running", stage="reviewer")
        validation_started = time.perf_counter()
        blocks = self._repo_memory(ctx.state)
        validation_payload = await self._critic.run_design_review(
            requirements=ctx.requirements,
            plan_content=updated_markdown,
            manifesto=ctx.state.manifesto,
            architecture=blocks.get("architecture", ""),
            design_record=json.dumps(self._design_record_payload(ctx.state.design_record) or {}, indent=2),
            modules=blocks.get("modules", ""),
            patterns=blocks.get("patterns", ""),
            constraints=ctx.state.constraints,
            prior_critique=ctx.issue_tracker.distilled_context(),
            model_override=ctx.selected_critic_model,
            session_id=ctx.session_id,
            round_number=ctx.round_number,
            mode="validation",
        )
        if not isinstance(validation_payload, ReviewResult):
            raise RuntimeError("Expected ReviewResult from validation phase")
        validation_review = validation_payload
        await emit_tool(
            "review_validation",
            "done",
            stage="reviewer",
            duration_ms=round((time.perf_counter() - validation_started) * 1000),
        )
        for issue_id in validation_review.resolved_issues:
            ctx.issue_tracker.resolve_issue(issue_id, ctx.round_number)
        validation_primary_id = ""
        if validation_review.primary_issue and validation_review.primary_issue.strip():
            validation_primary_id = ctx.issue_tracker.add_issue(
                validation_review.primary_issue,
                ctx.round_number,
                source="validation",
            ).id
        for item in [*validation_review.blocking_issues, *validation_review.architectural_concerns]:
            if item.strip():
                issue = ctx.issue_tracker.add_issue(item, ctx.round_number, source="validation")
                if validation_primary_id and issue.id:
                    ctx.issue_tracker.add_edge(validation_primary_id, issue.id, "causes")
        if self._causality_extractor is not None:
            self._causality_extractor.extract_edges(
                graph=ctx.issue_tracker,
                review=validation_review,
                round_number=ctx.round_number,
            )
        return validation_review

    async def convergence_check(
        self,
        *,
        ctx: PlanningRoundContext,
        updated_markdown: str,
        validation_review: ReviewResult,
        emit_tool: Callable[..., Any],
    ) -> tuple[ImplementabilityResult, ConvergenceResult]:
        implementability = ImplementabilityResult(
            implementable=True,
            missing_details=[],
            implementation_risks=[],
            suggested_additions=[],
            prose="Skipped - prerequisites not met",
        )
        score_history = [*ctx.state.review_score_history, float(validation_review.design_quality_score)]
        architecture_recent = ctx.state.architecture_change_rounds[-2:]
        architecture_stable = bool(architecture_recent) and not any(architecture_recent)
        score_stable = len(score_history) >= 3 and (max(score_history[-3:]) - min(score_history[-3:])) <= 0.6
        base_ready = (
            validation_review.review_complete
            and len(ctx.issue_tracker.root_open_issues()) == 0
            and int(ctx.issue_tracker.unresolved_dependency_chains()) == 0
            and validation_review.design_quality_score >= 8.5
            and ctx.round_number >= 2
        )
        stability_ready = architecture_stable and score_stable
        would_converge = base_ready and stability_ready
        if would_converge:
            await emit_tool("implementability_check", "running", stage="reviewer")
            impl_started = time.perf_counter()
            blocks = self._repo_memory(ctx.state)
            impl_payload = await self._critic.run_design_review(
                requirements=ctx.requirements,
                plan_content=updated_markdown,
                manifesto=ctx.state.manifesto,
                architecture=blocks.get("architecture", ""),
                design_record=json.dumps(self._design_record_payload(ctx.state.design_record) or {}, indent=2),
                modules=blocks.get("modules", ""),
                patterns=blocks.get("patterns", ""),
                constraints=ctx.state.constraints,
                prior_critique=ctx.issue_tracker.distilled_context(),
                model_override=ctx.selected_critic_model,
                session_id=ctx.session_id,
                round_number=ctx.round_number,
                mode="implementability",
            )
            if isinstance(impl_payload, ImplementabilityResult):
                implementability = impl_payload
            await emit_tool(
                "implementability_check",
                "done",
                stage="reviewer",
                duration_ms=round((time.perf_counter() - impl_started) * 1000),
            )
            if not implementability.implementable:
                for detail in implementability.missing_details:
                    ctx.issue_tracker.add_issue(detail, ctx.round_number)

        open_issue_count = len(ctx.issue_tracker.open_issues())
        root_open_issue_count = len(ctx.issue_tracker.root_open_issues())
        issue_history = [*ctx.state.open_issue_history, open_issue_count]
        recent_issue_history = issue_history[-3:]
        issue_trend_ready = len(recent_issue_history) >= 3 and all(
            recent_issue_history[idx] <= recent_issue_history[idx - 1] for idx in range(1, len(recent_issue_history))
        )
        ctx.state.review_score_history = score_history[-8:]
        ctx.state.open_issue_history = issue_history[-8:]
        converged = base_ready and stability_ready and issue_trend_ready and implementability.implementable
        convergence = ConvergenceResult(
            converged=converged,
            reason="review_complete" if converged else "review_open_issues",
            change_pct=0.0,
            regression=None,
            major_issues=root_open_issue_count,
        )
        if not converged and base_ready and root_open_issue_count == 0 and implementability.implementable:
            convergence.reason = "stability_not_met"
        return implementability, convergence
