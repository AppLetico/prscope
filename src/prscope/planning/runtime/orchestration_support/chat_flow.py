from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from typing import Any

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
                    core.save_plan_version(author_result.plan, round_number=0)
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
