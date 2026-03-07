from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable

from ....profile import build_profile
from ...core import PlanningCore
from ..author import AuthorResult

logger = logging.getLogger(__name__)


class RuntimeInitialDraftFlow:
    def __init__(
        self,
        runtime: Any,
        *,
        memory_prep_timeout_seconds: int,
        initial_draft_timeout_seconds: int,
    ):
        self._runtime = runtime
        self._memory_prep_timeout_seconds = memory_prep_timeout_seconds
        self._initial_draft_timeout_seconds = initial_draft_timeout_seconds

    async def prepare_memory(
        self,
        rebuild_memory: bool = False,
        progress_callback: Any = None,
        event_callback: Any = None,
    ) -> dict[str, str]:
        profile = build_profile(self._runtime.repo.resolved_path)
        await self._runtime.memory.ensure_memory(
            profile,
            rebuild=rebuild_memory,
            progress_callback=progress_callback,
            event_callback=event_callback,
        )
        blocks = self._runtime.memory.load_all_blocks()
        blocks["manifesto"] = self._runtime.memory.load_manifesto()
        return blocks

    async def run_initial_draft(
        self,
        core: PlanningCore,
        requirements: str,
        author_model_override: str | None = None,
        timeout_seconds_override: Callable[[], int] | int | None = None,
    ) -> AuthorResult:
        session = core.get_session()
        state = self._runtime._state(session.id, session)
        state.requirements = requirements
        blocks = self._runtime._repo_memory(state)
        manifesto = self._runtime._truncate_memory_block("manifesto", state.manifesto)
        skills_block = state.skills_context or self._runtime._skills_context(session.id)
        recall_block = self._runtime._build_recall_context(session.id, requirements)
        context_index = self._runtime._build_context_index(blocks)
        grounding_paths = set(self._runtime._session_reads(session.id)) | self._runtime.discovery.bootstrap_seed_paths(
            session.id
        )
        return await self._runtime.author.run_initial_draft(
            requirements=requirements,
            manifesto=manifesto,
            manifesto_path=self._runtime._manifesto_relative_path(),
            skills_block=skills_block,
            recall_block=recall_block,
            context_index=context_index,
            grounding_paths=grounding_paths,
            model_override=author_model_override,
            timeout_seconds_override=timeout_seconds_override,
        )

    async def continue_initial_draft(
        self,
        session_id: str,
        requirements: str,
        author_model_override: str | None = None,
        event_callback: Any | None = None,
        timeout_seconds: int | None = None,
        rebuild_memory: bool = False,
    ) -> Any:
        if timeout_seconds is None:
            timeout_seconds = getattr(
                self._runtime.planning_config,
                "initial_draft_timeout_seconds",
                self._initial_draft_timeout_seconds,
            )
        memory_timeout_seconds = getattr(
            self._runtime.planning_config,
            "memory_prep_timeout_seconds",
            self._memory_prep_timeout_seconds,
        )
        async with self._runtime._session_lock(session_id):
            core = self._runtime._core(session_id)
            session = core.get_session()
            self._runtime.tools.set_session(session.id)
            memory_cache_hit = False
            memory_elapsed = 0.0
            planner_elapsed = 0.0
            planner_started_at = 0.0
            failed_stage = "memory"

            async def wrapped_event(event: dict[str, Any]) -> None:
                await self._runtime._emit_event(event_callback, event, session_id)

            async def emit_draft_stage(stage: str, step: str, **extra: Any) -> None:
                await wrapped_event({"type": "setup_progress", "step": step, "draft_stage": stage, **extra})
                logger.info(
                    "initial_draft stage session_id=%s stage=%s step=%s extra=%s",
                    session_id,
                    stage,
                    step,
                    json.dumps(extra, default=str, sort_keys=True),
                )

            self._runtime.author.event_callback = wrapped_event
            snapshot = core.transition_and_snapshot(
                "draft",
                phase_message="Building initial plan draft...",
                pending_questions_json=None,
            )
            await wrapped_event(snapshot)
            await self._runtime._emit_event(
                event_callback,
                {"type": "thinking", "message": "Building initial plan draft..."},
                session_id,
            )
            await self._runtime._emit_event(
                event_callback,
                {"type": "phase_timing", "session_stage": "initial_draft", "state": "start"},
                session_id,
            )
            await wrapped_event({"type": "tool_call", "name": "draft_plan", "session_stage": "planner"})
            draft_started = time.perf_counter()
            await emit_draft_stage("memory", "Draft: checking codebase memory...")

            async def prepare_memory_only() -> None:
                nonlocal memory_cache_hit, memory_elapsed
                memory_started = time.perf_counter()

                async def on_memory_progress(step: str) -> None:
                    nonlocal memory_cache_hit
                    if step == "Using cached codebase memory.":
                        memory_cache_hit = True
                    await emit_draft_stage("memory", step)

                await self.prepare_memory(
                    rebuild_memory=rebuild_memory,
                    progress_callback=on_memory_progress,
                    event_callback=wrapped_event,
                )
                memory_elapsed = time.perf_counter() - memory_started

            async def run_planner_only() -> Any:
                nonlocal failed_stage, planner_elapsed, planner_started_at
                await self._runtime._emit_event(
                    event_callback,
                    {
                        "type": "thinking",
                        "message": f"Memory ready ({memory_elapsed:.1f}s). Calling author...",
                    },
                    session_id,
                )
                failed_stage = "planner"
                await emit_draft_stage(
                    "planner",
                    "Draft: starting planner pipeline...",
                    cache_hit=memory_cache_hit,
                    memory_elapsed_s=round(memory_elapsed, 3),
                )
                planner_started_at = time.perf_counter()
                result = await self.run_initial_draft(
                    core,
                    requirements=requirements,
                    author_model_override=self._runtime._resolve_author_model(
                        session,
                        author_model_override,
                    ),
                    timeout_seconds_override=lambda: max(
                        5,
                        int(max(5.0, float(timeout_seconds) - (time.perf_counter() - planner_started_at) - 1.0)),
                    ),
                )
                planner_elapsed = time.perf_counter() - planner_started_at
                await emit_draft_stage(
                    "planner_complete",
                    "Draft: planner pipeline complete.",
                    planner_elapsed_s=round(planner_elapsed, 3),
                )
                return result

            try:
                await asyncio.wait_for(prepare_memory_only(), timeout=memory_timeout_seconds)
                failed_stage = "planner"
                author_result = await asyncio.wait_for(run_planner_only(), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                elapsed = time.perf_counter() - draft_started
                if failed_stage == "memory":
                    error_msg = (
                        f"Memory preparation timed out after {memory_timeout_seconds}s ({elapsed:.1f}s elapsed)."
                    )
                else:
                    planner_timeout_elapsed = (
                        time.perf_counter() - planner_started_at if planner_started_at > 0 else 0.0
                    )
                    planner_elapsed = planner_timeout_elapsed
                    error_msg = (
                        f"Initial draft timed out after {timeout_seconds}s ({planner_timeout_elapsed:.1f}s elapsed)."
                    )
                await self._runtime._emit_event(event_callback, {"type": "error", "message": error_msg}, session_id)
                await self._runtime._emit_event(
                    event_callback,
                    {
                        "type": "phase_timing",
                        "session_stage": "initial_draft",
                        "state": "failed",
                        "elapsed_ms": round(elapsed * 1000.0, 2),
                        "failed_stage": failed_stage,
                    },
                    session_id,
                )
                await wrapped_event(
                    {
                        "type": "tool_result",
                        "name": "draft_plan",
                        "session_stage": "planner",
                        "duration_ms": round((time.perf_counter() - draft_started) * 1000),
                    }
                )
                snapshot = core.transition_and_snapshot("draft", phase_message=None)
                await self._runtime._emit_event(event_callback, snapshot, session_id)
                logger.error(
                    "initial_draft result=timeout session_id=%s failed_stage=%s cache_hit=%s "
                    "memory_elapsed_s=%.3f planner_elapsed_s=%.3f total_elapsed_s=%.3f",
                    session_id,
                    failed_stage,
                    memory_cache_hit,
                    memory_elapsed,
                    planner_elapsed,
                    elapsed,
                )
                raise RuntimeError(error_msg) from None
            except Exception:
                elapsed = time.perf_counter() - draft_started
                await self._runtime._emit_event(
                    event_callback,
                    {
                        "type": "phase_timing",
                        "session_stage": "initial_draft",
                        "state": "failed",
                        "elapsed_ms": round(elapsed * 1000.0, 2),
                        "failed_stage": failed_stage,
                    },
                    session_id,
                )
                await wrapped_event(
                    {
                        "type": "tool_result",
                        "name": "draft_plan",
                        "session_stage": "planner",
                        "duration_ms": round(elapsed * 1000),
                    }
                )
                snapshot = core.transition_and_snapshot("draft", phase_message=None)
                await self._runtime._emit_event(event_callback, snapshot, session_id)
                logger.exception(
                    "initial_draft result=error session_id=%s failed_stage=%s cache_hit=%s "
                    "memory_elapsed_s=%.3f planner_elapsed_s=%.3f total_elapsed_s=%.3f",
                    session_id,
                    failed_stage,
                    memory_cache_hit,
                    memory_elapsed,
                    planner_elapsed,
                    elapsed,
                )
                raise

            plan_content = author_result.plan
            self._runtime._record_session_reads(session.id, author_result.accessed_paths)
            self._runtime._state(session.id, session).design_record = self._runtime._design_record_from_payload(
                author_result.design_record
            )
            await wrapped_event(
                {
                    "type": "tool_result",
                    "name": "draft_plan",
                    "session_stage": "planner",
                    "duration_ms": round((time.perf_counter() - draft_started) * 1000),
                }
            )
            core.add_turn("author", self._runtime._author_chat_summary(plan_content, 0), round_number=0)
            plan_document = self._runtime._plan_document_from_version(plan_content, None)
            version = core.save_plan_version(plan_content, round_number=0, plan_document=plan_document)
            self._runtime._attach_plan_version_artifacts(
                version_id=version.id,
                plan_document=plan_document,
                plan_content=version.plan_content,
            )
            state = self._runtime._state(session.id, session)
            state.plan_markdown = version.plan_content
            self._runtime._persist_state_snapshot(session.id)
            logger.info(
                "initial_draft result=success session_id=%s cache_hit=%s memory_elapsed_s=%.3f "
                "planner_elapsed_s=%.3f total_elapsed_s=%.3f",
                session_id,
                memory_cache_hit,
                memory_elapsed,
                planner_elapsed,
                time.perf_counter() - draft_started,
            )
            await self._runtime._emit_event(
                event_callback,
                {
                    "type": "phase_timing",
                    "session_stage": "initial_draft",
                    "state": "complete",
                    "elapsed_ms": round((time.perf_counter() - draft_started) * 1000.0, 2),
                    "memory_elapsed_s": round(memory_elapsed, 3),
                    "planner_elapsed_s": round(planner_elapsed, 3),
                    "cache_hit": memory_cache_hit,
                },
                session_id,
            )
            await self._runtime._emit_event(
                event_callback,
                {
                    "type": "plan_ready",
                    "round": 0,
                    "initial_draft": True,
                    "saved_at_unix_s": time.time(),
                },
                session_id,
            )
            snapshot = core.transition_and_snapshot("refining", phase_message=None)
            await self._runtime._emit_event(event_callback, snapshot, session_id)
            await self._runtime._emit_event(
                event_callback,
                {"type": "complete", "message": "Initial draft complete"},
                session_id,
            )
            return self._runtime.store.get_planning_session(session_id) or session
