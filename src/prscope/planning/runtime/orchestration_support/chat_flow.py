from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import asdict
from typing import Any

from ..authoring.models import PLAN_SECTION_ORDER, apply_section_updates, render_markdown
from ..authoring.repair import extract_first_json_object, load_json_object
from ..context import ClarificationGate
from ..discovery import DiscoveryTurnResult
from ..followups import (
    apply_answer_to_graph,
    decision_graph_from_json,
    decision_graph_from_plan,
    decision_graph_to_json,
)
from ..reasoning import (
    IssueReferenceSignals,
    OpenQuestionResolutionSignals,
    ReasoningContext,
    RefinementMessageSignals,
    RefinementReasoner,
)


class RuntimeChatFlow:
    def __init__(self, runtime: Any):
        self._runtime = runtime
        self._reasoner = RefinementReasoner()

    def clarification_gate(self, session_id: str) -> ClarificationGate:
        gate = self._runtime._clarification_gates.get(session_id)
        if gate is None:
            gate = ClarificationGate(timeout_seconds=self._runtime.planning_config.clarification_timeout_seconds)
            self._runtime._clarification_gates[session_id] = gate
        return gate

    def provide_clarification(self, session_id: str, answers: list[str]) -> None:
        state = self._runtime._state(session_id)
        if answers:
            logs = state.clarification_logs
            pending_indices = [idx for idx, item in enumerate(logs) if not str(item.get("answer", "")).strip()]
            for idx, answer in zip(pending_indices, answers):
                logs[idx]["answer"] = answer
                logs[idx]["timed_out"] = False
            self._runtime.store.update_planning_session(
                session_id,
                clarifications_log_json=json.dumps(logs),
            )
        self.clarification_gate(session_id).provide_answer(answers)

    def abort_clarification(self, session_id: str) -> None:
        gate = self._runtime._clarification_gates.get(session_id)
        if gate is not None:
            gate.abort()

    @staticmethod
    def _open_questions_from_graph(graph: Any) -> str:
        unresolved = [
            node.description.strip() for node in graph.unresolved_nodes() if str(node.description or "").strip()
        ]
        if not unresolved:
            return "- None."
        return "\n".join(f"- {item}" for item in unresolved)

    @staticmethod
    def _classify_refinement_message_intent(user_message: str) -> str:
        return RefinementReasoner.classify_message_intent(user_message)

    @staticmethod
    def _heuristic_refinement_route(user_message: str) -> tuple[str | None, str, bool]:
        return RefinementReasoner.heuristic_route(user_message)

    @staticmethod
    def _is_small_refinement_request(user_message: str) -> bool:
        return RefinementReasoner.is_small_request(user_message)

    @classmethod
    def _extract_refinement_message_signals(
        cls,
        user_message: str,
        *,
        model_route: dict[str, str] | None = None,
    ) -> RefinementMessageSignals:
        return RefinementReasoner.extract_message_signals(user_message, model_route=model_route)

    async def _emit_routing_decision(
        self,
        *,
        event_callback: Any | None,
        session_id: str,
        route: str,
        source: str,
        confidence: str,
        message: str,
        evidence: list[str] | None = None,
        decision_source: str | None = None,
        reasoner_version: str | None = None,
    ) -> None:
        await self._runtime._emit_event(
            event_callback,
            {
                "type": "routing_decision",
                "session_stage": "refinement",
                "route": route,
                "source": source,
                "confidence": confidence,
                "message": message[:240],
                "evidence": list(evidence or [])[:6],
                "decision_source": decision_source,
                "reasoner_version": reasoner_version,
            },
            session_id,
        )

    async def _classify_ambiguous_refinement_message(
        self,
        *,
        session_id: str,
        user_message: str,
        current_plan_content: str,
        recent_turns: list[Any],
        author_model_override: str | None = None,
    ) -> dict[str, str] | None:
        prompt = self._reasoner.build_routing_prompt(current_plan_content, recent_turns, user_message)
        try:
            response, _ = await self._runtime.author._llm_call(  # noqa: SLF001
                [
                    {
                        "role": "system",
                        "content": "You classify refinement routing decisions. Return strict JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                allow_tools=False,
                max_output_tokens=400,
                model_override=author_model_override,
            )
        except Exception:
            return None
        raw = str(getattr(response.choices[0].message, "content", None) or "")
        try:
            json_text, _ = extract_first_json_object(raw)
            payload = load_json_object(json_text)
        except Exception:
            return None
        return self._reasoner.parse_routing_payload(payload)

    @staticmethod
    def _issue_match_tokens(text: str) -> set[str]:
        return RefinementReasoner.issue_match_tokens(text)

    def _resolve_targeted_lightweight_issue(
        self, *, user_message: str, round_number: int, tracker: Any | None
    ) -> str | None:
        if tracker is None or not hasattr(tracker, "open_issues") or not hasattr(tracker, "resolve_issue"):
            return None
        open_issues = [
            {"id": str(issue.id or "").strip(), "description": str(issue.description or "").strip()}
            for issue in list(tracker.open_issues())
        ]
        if not open_issues:
            return None
        decision = self._reasoner.resolve_issue_references(
            ReasoningContext(
                signals=IssueReferenceSignals(user_message=user_message, issues=open_issues),
                session_metadata={"scenario": "issue_resolution"},
            )
        )
        resolved_ids = list(decision.issue_resolution)
        if len(resolved_ids) != 1:
            return None
        tracker.resolve_issue(
            resolved_ids[0],
            round_number,
            propagate_causes=False,
            resolution_source="lightweight",
        )
        return resolved_ids[0]

    @staticmethod
    def _looks_like_open_question_answer(user_message: str) -> bool:
        return RefinementReasoner.looks_like_open_question_answer(user_message)

    @staticmethod
    def _looks_like_open_question_reopen(user_message: str) -> bool:
        return RefinementReasoner.looks_like_open_question_reopen(user_message)

    @staticmethod
    def _open_question_lines(raw: str) -> list[str]:
        lines = [line.strip() for line in str(raw).splitlines() if line.strip()]
        bullets = [line for line in lines if line.startswith("-") or line.startswith("*")]
        return bullets if bullets else lines

    @staticmethod
    def _message_explicitly_resolves_all_questions(user_message: str) -> bool:
        normalized = " ".join(user_message.lower().split())
        return any(
            token in normalized
            for token in ("all questions", "both questions", "resolve all", "no open questions", "none remaining")
        )

    def _guard_open_question_resolution(
        self,
        *,
        user_message: str,
        current_open_questions: str,
        proposed_open_questions: str | None,
    ) -> str | None:
        if proposed_open_questions is None:
            return None
        current_items = self._open_question_lines(current_open_questions)
        if len(current_items) <= 1 or self._message_explicitly_resolves_all_questions(user_message):
            return proposed_open_questions
        proposed_items = self._open_question_lines(proposed_open_questions)
        decision = self._reasoner.resolve_open_questions(
            ReasoningContext(
                signals=OpenQuestionResolutionSignals(
                    user_message=user_message,
                    current_items=current_items,
                    proposed_items=proposed_items,
                ),
                session_metadata={"scenario": "open_question_resolution"},
            )
        )
        return decision.resulting_open_questions or proposed_open_questions

    async def _apply_lightweight_plan_edit(
        self,
        *,
        session_id: str,
        user_message: str,
        author_model_override: str | None = None,
        event_callback: Any | None = None,
    ) -> None:
        async with self._runtime._session_lock(session_id):
            core = self._runtime._core(session_id)
            self._runtime.tools.set_session(session_id)
            session = core.get_session()
            if session.status not in {"refining", "converged"}:
                raise ValueError(f"Session is not in chat-refinement state: {session.status}")
            core.validate_command("message", session)
            if session.status == "converged":
                snapshot = core.transition_and_snapshot("refining", phase_message=None)
                await self._runtime._emit_event(event_callback, snapshot, session_id)
                session = core.get_session()

            current_plan = core.get_current_plan()
            if current_plan is None:
                raise ValueError("Cannot edit plan before an initial draft exists")
            round_number = session.current_round + 1
            state = self._runtime._state(session_id, session)
            selected_author_model = self._runtime._resolve_author_model(session, author_model_override)
            started = time.perf_counter()

            processing = core.transition_and_snapshot(
                "refining",
                phase_message="Applying lightweight edit",
                allow_round_stability=True,
            )
            await self._runtime._emit_event(event_callback, processing, session_id)
            await self._runtime._emit_event(
                event_callback,
                {
                    "type": "tool_update",
                    "tool": {
                        "name": "lightweight_edit",
                        "status": "running",
                        "session_stage": "author",
                        "query": user_message[:180],
                    },
                },
                session_id,
            )

            recent_turns = core.get_conversation()[-8:]
            history_lines = "\n".join(
                f"{turn.role}: {turn.content.strip()}" for turn in recent_turns if turn.content.strip()
            )
            current_plan_doc = self._runtime._plan_document_from_version(
                current_plan.plan_content,
                getattr(current_plan, "plan_json", None),
            )
            answering_open_question = self._looks_like_open_question_answer(user_message)
            reopening_open_question = self._looks_like_open_question_reopen(user_message)
            open_question_guidance = (
                "If the user message answers one or more items in open_questions, "
                "you MUST update `open_questions` to remove resolved questions "
                "(or replace with `- None.` when fully resolved), "
                "and reflect the decision in the relevant section(s)."
                if answering_open_question
                else (
                    "If the user is reopening or deferring a prior decision, update "
                    "`open_questions` to add or restore the unresolved question and "
                    "reflect the uncertainty in the relevant sections."
                )
                if reopening_open_question
                else "If the request affects open_questions, keep that section consistent with the new decisions."
            )
            prompt = (
                "Apply a small targeted edit request to the current plan.\n"
                "Make minimal changes and keep structure stable.\n"
                "Update at most 2 sections.\n"
                f"{open_question_guidance}\n"
                "Do NOT remove unrelated open questions. If only one question is answered, keep the others.\n"
                "Return strict JSON only with fields:\n"
                "- problem_understanding: str\n"
                "- updates: object {section_id: new_content}\n"
                "- assistant_reply: str\n"
                f"Allowed section_id values: {', '.join(PLAN_SECTION_ORDER)}."
            )
            response, _ = await self._runtime.author._llm_call(  # noqa: SLF001
                [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": (
                            f"## User request\n{user_message}\n\n"
                            f"## Recent conversation\n{history_lines}\n\n"
                            f"## Current plan JSON\n{json.dumps(current_plan_doc.__dict__, indent=2)}"
                        ),
                    },
                ],
                allow_tools=False,
                max_output_tokens=1400,
                model_override=selected_author_model,
            )
            raw = str(getattr(response.choices[0].message, "content", None) or "")
            json_text, _ = extract_first_json_object(raw)
            payload = load_json_object(json_text)
            updates_raw = payload.get("updates", {})
            if not isinstance(updates_raw, dict):
                raise ValueError("Lightweight edit output did not provide updates object")
            updates: dict[str, str] = {}
            for key, value in updates_raw.items():
                key_str = str(key).strip()
                if key_str in PLAN_SECTION_ORDER:
                    updates[key_str] = str(value)
            if "open_questions" in updates:
                updates["open_questions"] = (
                    self._guard_open_question_resolution(
                        user_message=user_message,
                        current_open_questions=current_plan_doc.open_questions,
                        proposed_open_questions=updates.get("open_questions"),
                    )
                    or updates["open_questions"]
                )
            updates = dict(list(updates.items())[:2])
            if not updates:
                raise ValueError("Lightweight edit produced no valid section updates")

            updated_plan = apply_section_updates(current_plan_doc, updates)
            changed_sections = sorted(
                section
                for section in updates
                if str(getattr(current_plan_doc, section, "")) != str(getattr(updated_plan, section, ""))
            )
            if not changed_sections:
                raise ValueError("Lightweight edit produced no material section changes")
            updated_markdown = render_markdown(updated_plan)

            core.add_turn("user", user_message, round_number=round_number)
            version = core.save_plan_version(
                updated_markdown,
                round_number=round_number,
                plan_document=updated_plan,
                changed_sections=changed_sections,
            )
            self._runtime._attach_plan_version_artifacts(
                version_id=version.id,
                plan_document=updated_plan,
                plan_content=updated_markdown,
                previous_graph_json=getattr(current_plan, "decision_graph_json", None),
            )
            state.revision_round = round_number
            state.plan_markdown = updated_markdown
            self._resolve_targeted_lightweight_issue(
                user_message=user_message,
                round_number=round_number,
                tracker=state.issue_tracker,
            )
            architecture_changed = any(section in {"architecture", "files_changed"} for section in changed_sections)
            state.architecture_change_count = int(state.architecture_change_count) + int(architecture_changed)
            state.architecture_change_rounds.append(architecture_changed)
            if len(state.architecture_change_rounds) > 8:
                state.architecture_change_rounds = state.architecture_change_rounds[-8:]
            reply = str(payload.get("assistant_reply", "")).strip() or (
                f"Applied your requested plan edit. Updated sections: {', '.join(changed_sections)}."
            )
            core.add_turn("author", reply, round_number=round_number)

            await self._runtime._emit_event(
                event_callback,
                {
                    "type": "tool_update",
                    "tool": {
                        "name": "lightweight_edit",
                        "status": "done",
                        "session_stage": "author",
                        "duration_ms": round((time.perf_counter() - started) * 1000),
                        "query": f"Updated: {', '.join(changed_sections)}",
                    },
                },
                session_id,
            )

            finalized = core.transition_and_snapshot(
                "refining",
                phase_message=None,
                current_round=round_number,
            )
            await self._runtime._emit_event(event_callback, finalized, session_id)
            # Persist snapshot before plan_ready so UI refreshes read the latest issue graph state.
            self._runtime._persist_state_snapshot(session_id)  # noqa: SLF001
            await self._runtime._emit_event(
                event_callback,
                {
                    "type": "plan_ready",
                    "round": round_number,
                    "saved_at_unix_s": time.time(),
                },
                session_id,
            )
            await self._runtime._emit_event(
                event_callback,
                {"type": "complete", "message": "Lightweight refinement complete"},
                session_id,
            )
            # Emit one final authoritative snapshot so UI gets completed tool groups
            # and clears any stale processing/activity indicators.
            final_sync = core.transition_and_snapshot(
                "refining",
                phase_message=None,
                allow_round_stability=True,
            )
            await self._runtime._emit_event(event_callback, final_sync, session_id)
            self._runtime._persist_state_snapshot(session_id)  # noqa: SLF001

    async def apply_followup_answer(
        self,
        *,
        session_id: str,
        followup_id: str,
        followup_answer: str,
        target_sections: list[str] | None = None,
        author_model_override: str | None = None,
        event_callback: Any | None = None,
    ) -> tuple[str, str | None]:
        async with self._runtime._session_lock(session_id):
            core = self._runtime._core(session_id)
            self._runtime.tools.set_session(session_id)
            session = core.get_session()
            if session.status not in {"refining", "converged"}:
                raise ValueError(f"Session is not in chat-refinement state: {session.status}")
            core.validate_command("followup_answer", session)
            if session.status == "converged":
                snapshot = core.transition_and_snapshot("refining", phase_message=None)
                await self._runtime._emit_event(event_callback, snapshot, session_id)
                session = core.get_session()

            current_plan = core.get_current_plan()
            if current_plan is None:
                raise ValueError("Cannot answer follow-up before an initial draft exists")

            round_number = session.current_round + 1
            state = self._runtime._state(session_id, session)
            selected_author_model = self._runtime._resolve_author_model(session, author_model_override)
            started = time.perf_counter()
            processing = core.transition_and_snapshot(
                "refining",
                phase_message="Applying follow-up decision",
                allow_round_stability=True,
            )
            await self._runtime._emit_event(event_callback, processing, session_id)
            await self._runtime._emit_event(
                event_callback,
                {
                    "type": "tool_update",
                    "tool": {
                        "name": "apply_followup_answer",
                        "status": "running",
                        "session_stage": "author",
                        "query": f"{followup_id} -> {followup_answer}"[:180],
                    },
                },
                session_id,
            )

            current_plan_doc = self._runtime._plan_document_from_version(
                current_plan.plan_content,
                getattr(current_plan, "plan_json", None),
            )
            current_graph = decision_graph_from_json(getattr(current_plan, "decision_graph_json", None))
            if not current_graph.nodes:
                current_graph = decision_graph_from_plan(
                    open_questions=current_plan_doc.open_questions,
                    plan_content=current_plan.plan_content,
                )
            updated_graph = apply_answer_to_graph(current_graph, followup_id, followup_answer)
            answered_node = updated_graph.nodes.get(followup_id)
            effective_sections = [
                section
                for section in (target_sections or [str(getattr(answered_node, "section", "") or "architecture")])
                if section in (*PLAN_SECTION_ORDER, "open_questions")
            ]
            if not effective_sections:
                effective_sections = ["architecture"]

            prompt = (
                "Apply a resolved follow-up decision to the current plan.\n"
                "The decision graph is already updated. Regenerate only the sections needed so the prose matches.\n"
                "Keep structure stable and return strict JSON only with fields:\n"
                "- problem_understanding: str\n"
                "- updates: object {section_id: new_content}\n"
                "- assistant_reply: str\n"
                f"Allowed section_id values: {', '.join(PLAN_SECTION_ORDER)}.\n"
                "At minimum, update the target sections listed by the user. Do not touch unrelated sections."
            )
            graph_payload = decision_graph_to_json(updated_graph)
            response, _ = await self._runtime.author._llm_call(  # noqa: SLF001
                [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": (
                            f"## Follow-up answer\n{followup_id}: {followup_answer}\n\n"
                            f"## Target sections\n{json.dumps(effective_sections)}\n\n"
                            f"## Updated decision graph JSON\n{graph_payload}\n\n"
                            f"## Current plan JSON\n{json.dumps(current_plan_doc.__dict__, indent=2)}"
                        ),
                    },
                ],
                allow_tools=False,
                max_output_tokens=1400,
                model_override=selected_author_model,
            )
            raw = str(getattr(response.choices[0].message, "content", None) or "")
            json_text, _ = extract_first_json_object(raw)
            payload = load_json_object(json_text)
            updates_raw = payload.get("updates", {})
            if not isinstance(updates_raw, dict):
                raise ValueError("Follow-up answer output did not provide updates object")
            updates: dict[str, str] = {}
            for key, value in updates_raw.items():
                key_str = str(key).strip()
                if key_str in PLAN_SECTION_ORDER:
                    updates[key_str] = str(value)
            updates["open_questions"] = self._open_questions_from_graph(updated_graph)

            updated_plan = apply_section_updates(current_plan_doc, updates)
            changed_sections = sorted(
                section
                for section in updates
                if str(getattr(current_plan_doc, section, "")) != str(getattr(updated_plan, section, ""))
            )
            if not changed_sections:
                raise ValueError("Follow-up answer produced no material section changes")

            updated_markdown = render_markdown(updated_plan)
            core.add_turn("user", f"Follow-up answer: {followup_id} -> {followup_answer}", round_number=round_number)
            version = core.save_plan_version(
                updated_markdown,
                round_number=round_number,
                plan_document=updated_plan,
                changed_sections=changed_sections,
            )
            self._runtime._attach_plan_version_artifacts(
                version_id=version.id,
                plan_document=updated_plan,
                plan_content=updated_markdown,
                previous_graph_json=graph_payload,
            )
            state.revision_round = round_number
            state.plan_markdown = updated_markdown
            architecture_changed = any(section in {"architecture", "files_changed"} for section in changed_sections)
            state.architecture_change_count = int(state.architecture_change_count) + int(architecture_changed)
            state.architecture_change_rounds.append(architecture_changed)
            if len(state.architecture_change_rounds) > 8:
                state.architecture_change_rounds = state.architecture_change_rounds[-8:]
            reply = str(payload.get("assistant_reply", "")).strip() or (
                f"Applied follow-up answer and refreshed: {', '.join(changed_sections)}."
            )
            core.add_turn("author", reply, round_number=round_number)

            await self._runtime._emit_event(
                event_callback,
                {
                    "type": "tool_update",
                    "tool": {
                        "name": "apply_followup_answer",
                        "status": "done",
                        "session_stage": "author",
                        "duration_ms": round((time.perf_counter() - started) * 1000),
                        "query": f"Updated: {', '.join(changed_sections)}",
                    },
                },
                session_id,
            )
            finalized = core.transition_and_snapshot(
                "refining",
                phase_message=None,
                current_round=round_number,
            )
            await self._runtime._emit_event(event_callback, finalized, session_id)
            self._runtime._persist_state_snapshot(session_id)  # noqa: SLF001
            await self._runtime._emit_event(
                event_callback,
                {
                    "type": "plan_ready",
                    "round": round_number,
                    "saved_at_unix_s": time.time(),
                },
                session_id,
            )
            await self._runtime._emit_event(
                event_callback,
                {"type": "complete", "message": "Follow-up decision applied"},
                session_id,
            )
            final_sync = core.transition_and_snapshot(
                "refining",
                phase_message=None,
                allow_round_stability=True,
            )
            await self._runtime._emit_event(event_callback, final_sync, session_id)
            self._runtime._persist_state_snapshot(session_id)  # noqa: SLF001
            return ("decision_refine", reply)

    async def handle_discovery_turn(
        self,
        session_id: str,
        user_message: str,
        author_model_override: str | None = None,
        critic_model_override: str | None = None,
        event_callback: Any | None = None,
    ) -> DiscoveryTurnResult:
        del critic_model_override
        async with self._runtime._session_lock(session_id):
            core = self._runtime._core(session_id)
            self._runtime.tools.set_session(session_id)
            session = core.get_session()
            if session.status != "draft":
                raise ValueError("Session is not in draft mode")
            core.validate_command("message", session)

            async def wrapped_event(event: dict[str, Any]) -> None:
                await self._runtime._emit_event(event_callback, event, session_id)

            self._runtime.discovery.event_callback = wrapped_event
            self._runtime.author.event_callback = wrapped_event

            current_round = session.current_round
            previous_pending_questions_json = session.pending_questions_json
            discovery_started = time.perf_counter()
            core.add_turn("user", user_message, round_number=current_round)
            processing_message = (
                "Analyzing your answers and preparing the first draft"
                if previous_pending_questions_json
                else "Analyzing your request and preparing clarifying questions"
            )
            snapshot = core.transition_and_snapshot(
                "draft",
                phase_message=processing_message,
                pending_questions_json=None,
            )
            await wrapped_event(snapshot)
            await wrapped_event({"type": "phase_timing", "session_stage": "discovery", "state": "start"})
            conversation = [{"role": turn.role, "content": turn.content} for turn in core.get_conversation()]
            discovery_context = "".join(
                [
                    self._runtime._skills_context(session_id),
                    self._runtime._build_recall_context(session_id, user_message),
                ]
            )
            try:
                selected_author_model = self._runtime._resolve_author_model(session, author_model_override)
                result = await self._runtime.discovery.handle_turn(
                    conversation,
                    session_id=session_id,
                    model_override=selected_author_model,
                    extra_context=discovery_context,
                )
                self._runtime._record_session_reads(session_id, self._runtime.tools.accessed_paths.copy())
                if not result.complete:
                    core.add_turn("author", result.reply, round_number=current_round)

                if result.complete:
                    summary = result.summary or user_message
                    self._runtime.store.update_planning_session(session_id, requirements=summary)
                    snapshot = core.transition_and_snapshot(
                        "draft",
                        phase_message="Building initial plan draft...",
                        pending_questions_json=None,
                    )
                    await wrapped_event(snapshot)
                    await wrapped_event(
                        {
                            "type": "phase_timing",
                            "session_stage": "discovery",
                            "state": "complete",
                            "elapsed_ms": round((time.perf_counter() - discovery_started) * 1000.0, 2),
                            "outcome": "draft",
                        }
                    )
                    author_result = await self._runtime._run_initial_draft(
                        core,
                        requirements=summary,
                        author_model_override=selected_author_model,
                    )
                    self._runtime._record_session_reads(session_id, author_result.accessed_paths)
                    self._runtime._state(session_id).design_record = self._runtime._design_record_from_payload(
                        author_result.design_record
                    )
                    core.add_turn(
                        "author",
                        self._runtime._author_chat_summary(author_result.plan, 0),
                        round_number=0,
                    )
                    plan_document = self._runtime._plan_document_from_version(author_result.plan, None)
                    version = core.save_plan_version(
                        author_result.plan,
                        round_number=0,
                        plan_document=plan_document,
                    )
                    self._runtime._attach_plan_version_artifacts(
                        version_id=version.id,
                        plan_document=plan_document,
                        plan_content=version.plan_content,
                    )
                    snapshot = core.transition_and_snapshot("refining", phase_message=None)
                    await wrapped_event(snapshot)
                else:
                    snapshot = core.transition_and_snapshot(
                        "draft",
                        phase_message=None,
                        pending_questions_json=(
                            json.dumps([asdict(question) for question in result.questions])
                            if result.questions
                            else None
                        ),
                    )
                    await wrapped_event(snapshot)
                    await wrapped_event(
                        {
                            "type": "phase_timing",
                            "session_stage": "discovery",
                            "state": "complete",
                            "elapsed_ms": round((time.perf_counter() - discovery_started) * 1000.0, 2),
                            "outcome": "questions" if result.questions else "message",
                        }
                    )
                await self._runtime._emit_event(
                    event_callback,
                    {"type": "complete", "message": "Discovery turn complete"},
                    session_id,
                )
                return result
            except Exception:
                latest = core.get_session()
                if latest.status == "draft" and bool(latest.is_processing):
                    recovery = core.transition_and_snapshot(
                        "draft",
                        phase_message=None,
                        pending_questions_json=previous_pending_questions_json,
                    )
                    await wrapped_event(recovery)
                await wrapped_event(
                    {
                        "type": "phase_timing",
                        "session_stage": "discovery",
                        "state": "failed",
                        "elapsed_ms": round((time.perf_counter() - discovery_started) * 1000.0, 2),
                    }
                )
                if event_callback:
                    maybe = event_callback({"type": "error", "message": "Discovery turn failed"})
                    if asyncio.iscoroutine(maybe):
                        await maybe
                raise

    async def chat_with_author(
        self,
        session_id: str,
        user_message: str,
        author_model_override: str | None = None,
        event_callback: Any | None = None,
    ) -> str:
        async with self._runtime._session_lock(session_id):
            core = self._runtime._core(session_id)
            session = core.get_session()
            if session.status not in {"refining", "converged"}:
                raise ValueError(f"Session is not in chat-refinement state: {session.status}")
            core.validate_command("message", session)

            status = session.status
            current_round = session.current_round
            current_plan = core.get_current_plan()
            if current_plan is None:
                raise ValueError("Cannot chat with author before an initial draft exists")

            snapshot = core.transition_and_snapshot(
                status,
                phase_message="Author is responding...",
                allow_round_stability=True,
            )
            await self._runtime._emit_event(event_callback, snapshot, session_id)

            completed = False
            try:
                core.add_turn("user", user_message, round_number=current_round)
                recent_turns = core.get_conversation()[-8:]
                history_lines = "\n".join(
                    f"{turn.role}: {turn.content.strip()}" for turn in recent_turns if turn.content.strip()
                )
                prompt = (
                    "You are the planning author in a refinement chat.\n"
                    "Answer the user's message conversationally and concisely.\n"
                    "Do not run critique yourself.\n"
                    "Do not claim the plan is updated unless a critique/refinement round is explicitly run.\n"
                    "If useful, propose what should be adjusted in the next critique run.\n\n"
                    f"Current plan (excerpt):\n{current_plan.plan_content[:3500]}\n\n"
                    f"Recent conversation:\n{history_lines}\n\n"
                    f"User message:\n{user_message}"
                )
                response, _ = await self._runtime.author._llm_call(  # noqa: SLF001
                    [
                        {"role": "system", "content": "You are a pragmatic software planning assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    allow_tools=False,
                    max_output_tokens=700,
                    model_override=self._runtime._resolve_author_model(session, author_model_override),
                )
                message = response.choices[0].message
                reply = str(getattr(message, "content", None) or "").strip()
                if not reply:
                    reply = (
                        "I hear you. I can discuss changes here, and when you're ready we can run "
                        "critique to apply updates to the plan."
                    )
                core.add_turn("author", reply, round_number=current_round)
                completed = True
                return reply
            finally:
                recovery = core.transition_and_snapshot(
                    status,
                    phase_message=None,
                    allow_round_stability=True,
                )
                await self._runtime._emit_event(event_callback, recovery, session_id)
                if completed:
                    await self._runtime._emit_event(
                        event_callback,
                        {"type": "complete", "message": "Author chat reply complete"},
                        session_id,
                    )

    async def handle_refinement_message(
        self,
        session_id: str,
        user_message: str,
        author_model_override: str | None = None,
        critic_model_override: str | None = None,
        event_callback: Any | None = None,
    ) -> tuple[str, str | None]:
        """Route refinement chat to either conversational reply or plan update round."""
        core = self._runtime._core(session_id)
        current_plan = core.get_current_plan()
        recent_turns = core.get_conversation()[-8:]
        model_route: dict[str, str] | None = None
        base_signals = self._extract_refinement_message_signals(user_message)
        if base_signals.heuristic_route is None and current_plan is not None:
            model_route = await self._classify_ambiguous_refinement_message(
                session_id=session_id,
                user_message=user_message,
                current_plan_content=current_plan.plan_content,
                recent_turns=recent_turns,
                author_model_override=author_model_override,
            )
        decision = await self._reasoner.decide(
            ReasoningContext(
                signals=self._extract_refinement_message_signals(user_message, model_route=model_route),
                plan_state=current_plan,
                session_metadata={"scenario": "route_message"},
            )
        )
        chosen_route = decision.route
        chosen_confidence = "high" if decision.confidence >= 0.8 else "medium" if decision.confidence >= 0.55 else "low"
        routing_source = "model" if any(item.startswith("model_route:") for item in decision.evidence) else (
            "heuristic" if base_signals.heuristic_route else "fallback"
        )
        if chosen_route == "lightweight_refine":
            await self._emit_routing_decision(
                event_callback=event_callback,
                session_id=session_id,
                route="lightweight_refine",
                source=routing_source,
                confidence=chosen_confidence,
                message=user_message,
                evidence=decision.evidence,
                decision_source=decision.decision_source,
                reasoner_version=decision.reasoner_version,
            )
            try:
                await self._apply_lightweight_plan_edit(
                    session_id=session_id,
                    user_message=user_message,
                    author_model_override=author_model_override,
                    event_callback=event_callback,
                )
            except Exception:
                await self._emit_routing_decision(
                    event_callback=event_callback,
                    session_id=session_id,
                    route="full_refine_after_lightweight_failure",
                    source="fallback",
                    confidence="medium",
                    message=user_message,
                    evidence=decision.evidence,
                    decision_source=decision.decision_source,
                    reasoner_version=decision.reasoner_version,
                )
                await self._runtime.run_adversarial_round(
                    session_id=session_id,
                    user_input=user_message,
                    author_model_override=author_model_override,
                    critic_model_override=critic_model_override,
                    event_callback=event_callback,
                )
            return ("refine_round", None)
        if chosen_route == "full_refine":
            await self._emit_routing_decision(
                event_callback=event_callback,
                session_id=session_id,
                route="full_refine",
                source=routing_source,
                confidence=chosen_confidence,
                message=user_message,
                evidence=decision.evidence,
                decision_source=decision.decision_source,
                reasoner_version=decision.reasoner_version,
            )
            await self._runtime.run_adversarial_round(
                session_id=session_id,
                user_input=user_message,
                author_model_override=author_model_override,
                critic_model_override=critic_model_override,
                event_callback=event_callback,
            )
            return ("refine_round", None)
        await self._emit_routing_decision(
            event_callback=event_callback,
            session_id=session_id,
            route="author_chat",
            source=routing_source,
            confidence=chosen_confidence,
            message=user_message,
            evidence=decision.evidence,
            decision_source=decision.decision_source,
            reasoner_version=decision.reasoner_version,
        )
        reply = await self.chat_with_author(
            session_id=session_id,
            user_message=user_message,
            author_model_override=author_model_override,
            event_callback=event_callback,
        )
        return ("author_chat", reply)
