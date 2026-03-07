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


class RuntimeChatFlow:
    def __init__(self, runtime: Any):
        self._runtime = runtime

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
    def _classify_refinement_message_intent(user_message: str) -> str:
        """Return 'refine' for plan-change instructions, otherwise 'chat'."""
        text = user_message.strip()
        if not text:
            return "chat"
        normalized = " ".join(text.lower().split())

        starts_like_question = bool(
            re.match(
                r"^(what|why|how|when|where|who|which|is|are|can|could|would|should|do|does|did)\b",
                normalized,
            )
        )
        has_question_mark = "?" in normalized
        if starts_like_question and has_question_mark:
            return "chat"

        explicit_refine_patterns = [
            r"\b(update|change|revise|modify|rewrite|adjust|fix|improve)\b.{0,40}\b(plan|draft|section|content)\b",
            r"\b(add|remove|replace|include|exclude)\b.{0,60}\b(plan|draft|section|"
            r"open questions?|architecture|requirements)\b",
            r"\b(we|you)\s+should\b",
            r"\bshould be\b",
            r"\bmake sure\b",
            r"\bdo not\b",
            r"\bdon't\b",
            r"\bplease\b.{0,40}\b(add|change|update|remove|replace|revise|modify|fix)\b",
            r"\blet'?s\b.{0,40}\b(add|change|update|remove|replace|revise|modify|fix)\b",
            r"^(yes|no)\b",
        ]
        if any(re.search(pattern, normalized) for pattern in explicit_refine_patterns):
            return "refine"

        # Imperative-style statements are usually intended as plan edits.
        if re.search(r"^(add|remove|replace|update|change|revise|modify|rewrite|fix|include|exclude)\b", normalized):
            return "refine"

        return "chat"

    @staticmethod
    def _is_small_refinement_request(user_message: str) -> bool:
        normalized = " ".join(user_message.lower().split())
        if not normalized:
            return False
        if len(normalized) > 280:
            return False
        broad_change_signals = [
            "architecture",
            "system design",
            "refactor",
            "migration",
            "database",
            "auth",
            "authentication",
            "authorization",
            "security",
            "new service",
            "new module",
            "across all endpoints",
            "across the codebase",
            "multiple files",
            "end-to-end",
            "rewrite the plan",
            "overhaul",
        ]
        if any(signal in normalized for signal in broad_change_signals):
            return False
        small_change_signals = [
            "you should",
            "we should",
            "should be",
            "log",
            "verify",
            "validation",
            "test strategy",
            "test coverage",
            "monitor",
            "monitoring",
            "observability",
            "rollback",
            "roll back",
            "error handling",
            "guardrail",
            "guardrails",
            "wording",
            "clarify",
            "open question",
            "summary",
            "non-goal",
            "non goal",
            "files changed",
            "section",
            "rename",
            "typo",
            "small change",
        ]
        return any(signal in normalized for signal in small_change_signals)

    @staticmethod
    def _issue_match_tokens(text: str) -> set[str]:
        stopwords = {
            "please",
            "update",
            "plan",
            "address",
            "these",
            "review",
            "notes",
            "note",
            "start",
            "here",
            "with",
            "without",
            "should",
            "would",
            "could",
            "need",
            "needed",
            "include",
            "adjust",
            "approach",
            "task",
            "tasks",
            "success",
            "where",
            "when",
            "into",
            "from",
        }
        tokens: set[str] = set()
        for raw in re.findall(r"[a-z0-9_]+", text.lower()):
            token = raw
            if token.startswith("issue_"):
                tokens.add(token)
                continue
            if token.endswith("ies") and len(token) > 4:
                token = f"{token[:-3]}y"
            elif token.endswith("es") and len(token) > 4:
                token = token[:-2]
            elif token.endswith("s") and len(token) > 4:
                token = token[:-1]
            if len(token) < 4 or token in stopwords:
                continue
            tokens.add(token)
        return tokens

    def _resolve_targeted_lightweight_issue(
        self, *, user_message: str, round_number: int, tracker: Any | None
    ) -> str | None:
        if tracker is None or not hasattr(tracker, "open_issues") or not hasattr(tracker, "resolve_issue"):
            return None
        open_issues = list(tracker.open_issues())
        if not open_issues:
            return None

        normalized_message = " ".join(user_message.lower().split())
        explicit_matches = [issue.id for issue in open_issues if issue.id and issue.id.lower() in normalized_message]
        if len(explicit_matches) == 1:
            tracker.resolve_issue(
                explicit_matches[0],
                round_number,
                propagate_causes=False,
                resolution_source="lightweight",
            )
            return explicit_matches[0]

        exact_description_matches = [
            issue.id
            for issue in open_issues
            if issue.description and " ".join(issue.description.lower().split()) in normalized_message
        ]
        if len(exact_description_matches) == 1:
            tracker.resolve_issue(
                exact_description_matches[0],
                round_number,
                propagate_causes=False,
                resolution_source="lightweight",
            )
            return exact_description_matches[0]

        message_tokens = self._issue_match_tokens(user_message)
        if not message_tokens:
            return None

        scored: list[tuple[float, str]] = []
        for issue in open_issues:
            issue_tokens = self._issue_match_tokens(issue.description)
            if not issue_tokens:
                continue
            overlap = len(issue_tokens & message_tokens) / len(issue_tokens)
            if overlap > 0:
                scored.append((overlap, issue.id))
        if not scored:
            return None

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_issue_id = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        if best_score >= 0.45 and (len(scored) == 1 or (best_score - second_score) >= 0.15):
            tracker.resolve_issue(
                best_issue_id,
                round_number,
                propagate_causes=False,
                resolution_source="lightweight",
            )
            return best_issue_id
        return None

    @staticmethod
    def _looks_like_open_question_answer(user_message: str) -> bool:
        normalized = " ".join(user_message.lower().split())
        if not normalized:
            return False
        if "?" in normalized:
            return False
        answer_like_starts = ("yes", "no", "it should", "we should", "should be", "prefer", "i prefer")
        if normalized.startswith(answer_like_starts):
            return True
        return any(token in normalized for token in ["should", "must", "prefer", "we'll", "we will"])

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
        """Avoid over-clearing open questions for single-answer messages."""
        if proposed_open_questions is None:
            return None
        current_items = self._open_question_lines(current_open_questions)
        if len(current_items) <= 1:
            return proposed_open_questions
        if self._message_explicitly_resolves_all_questions(user_message):
            return proposed_open_questions

        proposed_items = self._open_question_lines(proposed_open_questions)
        proposed_is_none = not proposed_items or proposed_open_questions.strip().lower() in {"none", "- none.", "none."}
        removed_too_many = proposed_is_none or len(proposed_items) < (len(current_items) - 1)
        if removed_too_many and self._looks_like_open_question_answer(user_message):
            # Conservatively resolve only the first outstanding item unless user said all.
            return "\n".join(current_items[1:])
        return proposed_open_questions

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
            open_question_guidance = (
                "If the user message answers one or more items in open_questions, "
                "you MUST update `open_questions` to remove resolved questions "
                "(or replace with `- None.` when fully resolved), "
                "and reflect the decision in the relevant section(s)."
                if answering_open_question
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
        intent = self._classify_refinement_message_intent(user_message)
        if intent == "refine":
            if self._is_small_refinement_request(user_message):
                try:
                    await self._apply_lightweight_plan_edit(
                        session_id=session_id,
                        user_message=user_message,
                        author_model_override=author_model_override,
                        event_callback=event_callback,
                    )
                except Exception:
                    await self._runtime.run_adversarial_round(
                        session_id=session_id,
                        user_input=user_message,
                        author_model_override=author_model_override,
                        critic_model_override=critic_model_override,
                        event_callback=event_callback,
                    )
            else:
                await self._runtime.run_adversarial_round(
                    session_id=session_id,
                    user_input=user_message,
                    author_model_override=author_model_override,
                    critic_model_override=critic_model_override,
                    event_callback=event_callback,
                )
            return ("refine_round", None)
        reply = await self.chat_with_author(
            session_id=session_id,
            user_message=user_message,
            author_model_override=author_model_override,
            event_callback=event_callback,
        )
        return ("author_chat", reply)
